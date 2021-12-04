import importlib
import model.trainer
import data.VCTK
import torch
import signal

if __name__ == "__main__":
    dataset = data.VCTK.VCTKDataset(
        text_file_paths=["resources/tomscott/txt/tomscott.txt"],
        audio_file_paths=["resources/tomscott/wav48/tomscott.wav"]
    )
    print(f"Loaded {len(dataset)} entries")

    trainer = model.trainer.Trainer(
        device=torch.device("cuda"),
        checkpoint=300802,
        checkpoint_dir="checkpoints/tom",
        load_from_checkpoint=True,
    )

    # allow graceful exit
    def exit(*args):
        trainer.save()
    signal.signal(signal.SIGINT, exit)
    signal.signal(signal.SIGTERM, exit)

    try:
        trainer.train(
            epochs=1000,
            dataset=dataset,
            save_every_n=50,
            batch_size=4,
            run_name="runs/tom2"
        )
    except Exception as e:
        x = trainer.checkpoint
        trainer.checkpoint=str(trainer.checkpoint)+"failure"
        trainer.save()
        trainer.checkpoint = x+1
        print(str(e))

    trainer.checkpoint+=1
    trainer.save()