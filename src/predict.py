import cv2 as cv
from ultralytics import YOLO
import src.config as CONFIG
import src.constants as CONSTANTS
import src.helper as helper

model = YOLO(f"models/{CONFIG.MODEL_NAME}")

results = model.track("data/raw/raw_video1.mp4")

print("Processing Completed")
print(CONSTANTS.DIV)

first_result = results[0]
height, width = first_result.orig_img.shape[:2]

fps = 30
fourcc = cv.VideoWriter_fourcc(*'mp4v')

out = cv.VideoWriter(
    "output/tracked_video.mp4",
    fourcc,
    fps,
    (width, height)
)

for result in results:
    img = result.orig_img.copy()  

    for box in result.boxes:
        p1 = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))
        p2 = (int(box.xyxy[0][2]), int(box.xyxy[0][3]))

        coords = helper.get_coords(p1, p2)
        text = result.names[int(box.cls)]

        cv.line(img, coords[0], coords[1], CONSTANTS.LABEL.RED, 2)
        cv.line(img, coords[1], coords[2], CONSTANTS.LABEL.RED, 2)
        cv.line(img, coords[2], coords[3], CONSTANTS.LABEL.RED, 2)
        cv.line(img, coords[3], coords[0], CONSTANTS.LABEL.RED, 2)

        cv.putText(
            img,
            text,
            (coords[0][0], coords[0][1] - 10),
            CONSTANTS.LABEL.FONT,
            1.0,
            CONSTANTS.LABEL.RED,
            2
        )

    out.write(img)
    cv.imshow("Tracking", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv.destroyAllWindows()

print("Video saved as output/tracked_video.mp4")