import model.trainer
import torch
import signal

if __name__ == "__main__":
    trainer = model.trainer.Trainer(
        device=torch.device("cuda"),
        checkpoint=300051,
        load_from_checkpoint=True,
    )
    trainer.checkpoint = 300055
    # allow graceful exit
    def exit(*args):
        trainer.save()
    signal.signal(signal.SIGINT, exit)
    signal.signal(signal.SIGTERM, exit)

    try:
        trainer.train(
            epochs=500,
            save_every_n=50,
            batch_size=4,
            run_name="runs/day2",
        )
    except Exception as e:
        x = trainer.checkpoint
        trainer.checkpoint=str(trainer.checkpoint)+"failure"
        trainer.save()
        trainer.checkpoint = x+1
        print(str(e))