from training.data_loader import load_datasets
from training.model_builder import build_model
from training.callbacks import get_callbacks

from config import MODEL_PATH, EPOCHS

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

train_ds, val_ds, class_names = load_datasets()

print(f"\nClasses : {len(class_names)}")

print("\nBuilding Model...")

model = build_model(len(class_names))

model.summary()

print("\nStarting Training...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=get_callbacks(MODEL_PATH)
)

print("\nSaving Final Model...")

model.save(MODEL_PATH)

print("\nTraining Completed!")
print(f"Model Saved At : {MODEL_PATH}")