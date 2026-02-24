# Callbacks

[Callbacks](./main_classes/callback) are another way to customize [`Trainer`], but they don't change anything *inside the training loop*. Instead, a callback inspects the training loop state and executes some action (early stopping, logging, etc.) depending on the state. For example, you can't implement a custom loss function with a callback because that requires overriding [`~Trainer.compute_loss`].

To use a callback, create a class that inherits from [`TrainerCallback`] and implements the functionality you want. Then pass the callback to the `callback` parameter in [`Trainer`]. The example below implements an early stopping callback that stops training after 10 steps.

```py
from transformers import TrainerCallback, Trainer

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, num_steps=10):
        self.num_steps = num_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.num_steps:
            return {"should_training_stop": True}
        else:
            return {}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback()],
)
```
