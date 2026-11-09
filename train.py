from fastai.vision.all import *
from fastseg import MobileV3Small
import matplotlib.pyplot as plt
from datetime import datetime
import config
from support import metrics, data
from support.loss import HybridTverskyCELoss

if __name__ == "__main__":
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = config.MODEL_PATH / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.txt", "w") as saved:
        saved.write(f"epochs={config.EPOCHS}\n")
        saved.write(f"batch_size={config.BATCH_SIZE}\n")
        saved.write(f"ce_weight={config.CE_WEIGHT}")
        saved.write(f"tversky_a={config.TVERSKY_A}")
        saved.write(f"tversky_b={config.TVERSKY_B}")

    model = MobileV3Small.from_pretrained()
    model.last = nn.Conv2d(model.last.in_channels, len(config.CODES), kernel_size=1)
    model = model.cuda()

    dls = data.make_dls()

    if config.DEBUG == True:
        dls.show_batch(max_n=12, figsize=(10,7))

        print("Training:", len(dls.train_ds))
        print("Validation:", len(dls.valid_ds))

        plt.savefig(run_dir / "preview.png")
        plt.close()

    learn = Learner(
        dls, 
        model, 
        loss_func=HybridTverskyCELoss(), 
        metrics=[
            DiceMulti(), 
            AccumMetric(metrics.precision, flatten=False), 
            AccumMetric(metrics.recall, flatten=False), 
            AccumMetric(metrics.left_precision, flatten=False), 
            AccumMetric(metrics.left_recall, flatten=False), 
            AccumMetric(metrics.right_precision, flatten=False), 
            AccumMetric(metrics.right_recall, flatten=False)
        ]
    )

    if config.EPOCHS > 0:
        learn.fine_tune(config.EPOCHS)

    learn.metrics = []
    learn.export(run_dir / "lane_model.pkl")
    learn.show_results(max_n=12, figsize=(10,7))
    plt.savefig(run_dir / "results.png")
    plt.close()
