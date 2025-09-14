if batch_idx == 0:
    # DECODE

    # y_trues_decoded_yolo_format: (bs, S * S, 6) -> 6 is [class, obj_conf, x, y, w, w]
    # y_trues_decoded_voc_format: (bs, S * S, 4)
    # y_preds_decoded_yolo_format: (bs, S * S, 6)
    # y_preds_decoded_voc_format: (bs, S * S, 4)

    y_trues_decoded_yolo_format = decode(y_trues.detach().cpu())

    # FIXME: here uses yolo2voc which is mutable and hence
    # if I print y_trues_decoded_yolo_format, y_preds_decoded_voc_format
    # they both the same after.
    # note yolo2voc expects [x, y, w, h] format so need slice
    y_trues_decoded_voc_format = yolo2voc(
        y_trues_decoded_yolo_format[..., 2:], height=448, width=448
    )

    y_preds_decoded_yolo_format = decode(y_preds.detach().cpu())
    y_preds_decoded_voc_format = yolo2voc(
        y_preds_decoded_yolo_format[..., 2:], height=448, width=448
    )

    inputs = inputs.detach().cpu()
    image_grid = []
    # TODO: HONGNAN: remember find a way to turn TOTENSOR back to proper uint8 image? how?
    for (
        input,
        y_true_decoded_voc_format,
        y_pred_decoded_yolo_format,
        y_pred_decoded_voc_format,
    ) in zip(
        inputs,
        y_trues_decoded_voc_format,
        y_preds_decoded_yolo_format,
        y_preds_decoded_voc_format,
    ):
        # FIXME: find way to turn Tensor back to uint8 image without using this.
        # input: (3, 448, 448)
        input = torch.from_numpy(np.asarray(FT.to_pil_image(input))).permute(
            2, 0, 1
        )

        # nms_bbox_pred: (N, 6) where N is number of bboxes after nms
        #                       and 6 is [class, obj_conf, x, y, w, h]
        nms_bbox_pred = non_max_suppression(
            y_pred_decoded_yolo_format,
            iou_threshold=0.5,
            obj_threshold=0.4,
            bbox_format="yolo",
        )

        num_bboxes_after_nms = nms_bbox_pred.shape[0]

        if num_bboxes_after_nms == 0:
            # if no bboxes after nms, then just plot the image
            # but in order for consistency, we just pass empty colors
            # and class names to torchvision.utils.draw_bounding_boxes
            # so it can just plot the original image instead of using continue.
            colors = []
            class_names = []
        else:
            class_names = [
                ClassMap.classes_map[int(class_idx.item())]
                for class_idx in nms_bbox_pred[:, 0]
            ]
            colors = ["red"] * num_bboxes_after_nms

        font_path = "./07558_CenturyGothic.ttf"

        # num_colors =
        # print(f"nms: {nms_bbox_pred[...,2:]}")
        overlayed_image_true = torchvision.utils.draw_bounding_boxes(
            input,
            y_true_decoded_voc_format,
            # colors=colors,
            width=6,
            # labels=class_names,
        )
        overlayed_image_pred = torchvision.utils.draw_bounding_boxes(
            input,
            nms_bbox_pred[..., 2:],
            colors=colors,
            width=6,
            labels=class_names,
            font_size=20,
            font=font_path,
        )
        image_grid.append(overlayed_image_true)
        image_grid.append(overlayed_image_pred)
    grid = torchvision.utils.make_grid(image_grid)

    # print(f"shape of overlayed_images: {overlayed_images.shape}")
    fig = plt.figure(figsize=(30, 30))
    plt.imshow(grid.numpy().transpose(1, 2, 0))

    plt.savefig(f"epoch_{epoch}_batch0.png", dpi=300)
    # plt.show()

    # END DECODE


def postprocess(nms: bool = True):
    pass