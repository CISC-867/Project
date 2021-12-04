import model.trainer

if __name__ == "__main__":
    trainer = model.trainer.Trainer(
        checkpoint=2,
        load_from_checkpoint=True
    )
    trainer.train()