import model.trainer

if __name__ == "__main__":
    trainer = model.trainer.Trainer(
        checkpoint=3,
        load_from_checkpoint=False
    )
    trainer.train()