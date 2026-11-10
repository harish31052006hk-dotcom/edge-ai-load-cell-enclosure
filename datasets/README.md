# Datasets & Physical Test Samples

This directory holds the raw photographic test samples used to validate the Edge AI Load Cell Enclosure detector.

## Physical Specifications of Target Object
- **Target Component:** Custom 3D-printed / machined Load Cell Enclosure.
- **Nominal Dimensions:** $32.0\text{ mm} \times 32.0\text{ mm}$ (square footprint).
- **Inspection Classes:**
  1. `bottom`: Lower enclosure base holding the load cell core.
  2. `top`: Upper protective cover.
  3. `enclosure_assembled`: Mated assembly of top cover and bottom base.

## Available Physical Test Images

| File Name | Target Class | Resolution | Description |
| :--- | :--- | :--- | :--- |
| [`test_image.jpg`](./test_image.jpg) | `enclosure_assembled` | Full resolution photograph | Assembled load cell enclosure unit under evaluation lighting. |
| [`bottom.png`](./bottom.png) | `bottom` | Full resolution photograph | Separate lower base component showing cavity and mounting geometry. |
| [`top.png`](./top.png) | `top` | Full resolution photograph | Separate upper cover component. |
| [`top and bottom.png`](./top and bottom.png) | `top` & `bottom` | Full resolution photograph | Dual-component scene testing simultaneous independent classification. |

## Dataset Split & Training Lineage
- **Full Training Dataset:** Annotated and hosted remotely via **Roboflow**.
- **Dataset Split (Train / Val / Test):** *Not completed / Not available in local directory* (split management and image augmentation were managed remotely inside the Roboflow cloud pipeline).

