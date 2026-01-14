import numpy as np
import torch
import cv2
from PIL import Image
from torchvision.ops import box_convert
import supervision as sv
from Grounded_SAM_2.grounding_dino.groundingdino.util.inference import predict
import Grounded_SAM_2.grounding_dino.groundingdino.datasets.transforms as T
from Grounded_SAM_2.utils.track_utils import sample_points_from_masks

if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
transform = T.Compose([
        T.RandomResize([640], max_size=1000),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

last_boxes = None
torch.cuda.synchronize()

@torch.no_grad()
def run_realtime_sam(
    frame_generator,
    grounding_model,
    image_predictor,
    video_predictor,
    text_prompt="bottle.",
    device="cuda",
    box_threshold=0.35,
    text_threshold=0.25,
    prompt_type="box",
    inference_interval=30
):
  
    inference_state = video_predictor.init_state(video_path=None)
    print("[INFO] Initialized video inference state.")
    empty_points = torch.empty((0, 2), device=device, dtype=torch.float32)
    empty_labels = torch.empty((0), device=device, dtype=torch.int32)

    for frame_idx, frame in enumerate(frame_generator):
        if inference_state["images"] is None:
            inference_state["images"] = [] 
            

        current_frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).to(device).half() / 255.0
        
     
        current_frame_tensor = torch.nn.functional.interpolate(
            current_frame_tensor.unsqueeze(0), 
            size=(1024, 1024), 
            mode="bilinear"
        ).squeeze(0)

        if frame_idx == 0:
            h, w, _ = frame.shape
            inference_state["video_height"] = h
            inference_state["video_width"] = w
            inference_state["images"] = [None] * 100000 
            
        inference_state["images"][frame_idx] = current_frame_tensor.to(device)
        run_new_detection = (frame_idx % inference_interval == 0)
        

        if run_new_detection:
            video_predictor.reset_state(inference_state)
            inference_state["images"] = [None] * (frame_idx + inference_interval + 1)
            inference_state["images"][frame_idx] = current_frame_tensor.to(device)


            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            image_for_model, _ = transform(image_pil, None)

            with torch.autocast(device_type=device, dtype=torch.float16):
                print("running grounded dino inference...")
                boxes, confidences, labels = predict(
                    model=grounding_model,
                    image=image_for_model,
                    caption=text_prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    device=device
                )
            print(f"[INFO] Frame {frame_idx}: Detected {len(boxes)} objects with Grounded DINO for prompt '{text_prompt}' label {labels}.")

            h, w, _ = frame.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            last_boxes = input_boxes

            if prompt_type == "box":
                for obj_id, box in enumerate(input_boxes, start=1):
                    print(f"DEBUG: Frame {frame_idx}, Obj {obj_id}, Points Type: {type(empty_points)}, Box: {box}")

                    video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        box=box,
                        points=empty_points, 
                        labels=empty_labels  
                    )
            
            elif prompt_type in ["mask", "point"]:
                image_predictor.set_image(frame)
                print("running sam2 image predictor inference...")
                masks, _, _ = image_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False
                )
                if masks.ndim == 4: masks = masks.squeeze(1)

                if prompt_type == "mask":
                    for obj_id, mask in enumerate(masks, start=1):
                        video_predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=frame_idx,
                            obj_id=obj_id,
                            mask=mask
                        )
                elif prompt_type == "point":
                    all_points = sample_points_from_masks(masks=masks, num_points=10)
                    for obj_id, points in enumerate(all_points, start=1):
                        labels_array = np.ones(points.shape[0], dtype=np.int32)
                        video_predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=frame_idx,
                            obj_id=obj_id,
                            points=points,
                            labels=labels_array
                        )


        video_segments = {}
        if run_new_detection and input_boxes is not None and len(input_boxes) > 0:

            image_predictor.set_image(frame)
            print("running sam2 image predictor inference for new detection frame...")
            masks, scores, _ = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False
            )

            video_segments = {i+1: masks[i] for i in range(len(masks))}

        elif len(inference_state.get("obj_id_to_idx", {})) > 0:
            print(f"[INFO] Frame {frame_idx}: Propagating masks for existing objects.")
            print(f"DEBUG: inference_state obj_id_to_idx keys: {list(inference_state.get('obj_id_to_idx', {}).keys())}")
            try:
                generator = video_predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=frame_idx-1,
                    max_frame_num_to_track=30
                )
                res = next(generator, None)
                if res is not None:
                    print(f"[INFO] Frame {frame_idx}: Propagated masks for existing objects.")
                    _, out_obj_ids, out_mask_logits = res
                    
                    video_segments = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
            
            except Exception as e:
                print(f"Tracking error at frame {frame_idx}: {e}")
        if len(video_segments) > 0:
            object_ids = list(video_segments.keys())
            masks_list = list(video_segments.values())

            processed_masks = []
            for m in masks_list:
                if m.ndim == 3:
                    m = m.squeeze(0) 
                processed_masks.append(m.astype(bool)) 
            
            masks_array = np.stack(processed_masks)

            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks_array),
                mask=masks_array,
                class_id=np.array(object_ids, dtype=np.int32)
            )
            
            annotated_frame = frame.copy()
            annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections=detections)
            annotated_frame = sv.MaskAnnotator().annotate(annotated_frame, detections=detections)
                
        else:
            annotated_frame = frame.copy()
        yield annotated_frame