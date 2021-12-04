import model.trainer
import torch

if __name__ == "__main__":
    trainer = model.trainer.Trainer(
        device=torch.device("cuda"),
        checkpoint=7,
        load_from_checkpoint=True,
    )
    try:
        trainer.train(
            epochs=500,
            save_every_n=50,
            batch_size=4,
            run_name="runs/overnight1",
        )
    except Exception:
        trainer.checkpoint=trainer.checkpoint+"failure"
        trainer.save()