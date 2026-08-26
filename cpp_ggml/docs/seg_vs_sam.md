# Instance Segmentation and SAM

YOLO `segment` and SAM solve different input/output contracts:

| Model family | Input contract | Output contract | Best use |
|---|---|---|---|
| YOLO-seg / YOLOE-seg | One image, trained category vocabulary or text prompts | Boxes, class scores, and one mask per detected instance | Fast, repeatable category-aware video and batch inference |
| SAM / MobileSAM | Image plus a point, box, or mask prompt | A mask for the prompted object, without a fixed category vocabulary | Interactive annotation, unknown objects, and refinement workflows |

YOLO segmentation answers “which known or prompted categories are present and
where are their instances?” SAM answers “which pixels belong to this prompted
region?” A YOLO mask is produced together with detection confidence and class
identity; a SAM mask has no class identity unless an application adds a separate
classifier. Use YOLO-seg when throughput and class filtering matter, and SAM when
the user supplies the object prompt or the category is not known in advance.
