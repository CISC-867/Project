import model.trainer

if __name__ == "__main__":
    trainer = model.trainer.Trainer(
        checkpoint=5,
        load_from_checkpoint=False
    )
    trainer.train(save_every_n=100)