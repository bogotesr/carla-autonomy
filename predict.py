from fastai.vision.all import *
from pathlib import Path
import matplotlib.pyplot as plt
import config

if __name__ == "__main__":
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = config.PREDICTION_PATH / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    learner = load_learner(config.LEARNER_PATH)
    photos = get_image_files(config.INPUTS_PATH)

    for photo in photos:
        img = PILImage.create(photo)
        prediction, prediction_idx, prob = learner.predict(img)
        fig, ax = plt.subplots(1, 5, figsize=(60,10))

        prediction_idx = torch.argmax(prediction_idx, dim=0)
        prediction_idx = prediction_idx.detach().cpu().numpy()

        ax[0].imshow(img)
        ax[0].set_title("Input")

        ax[1].imshow(prediction_idx)
        ax[1].set_title("Prediction")

        ax[2].imshow(prob[0], vmin=0, vmax=1)
        ax[2].set_title("Background")
        
        ax[3].imshow(prob[1], vmin=0, vmax=1)
        ax[3].set_title("Left Boundary")
        
        ax[4].imshow(prob[2], vmin=0, vmax=1)
        ax[4].set_title("Right Boundary")

        plt.tight_layout()
        plt.savefig(run_dir / f"{photo.stem}_prediction.png")
        plt.close()
