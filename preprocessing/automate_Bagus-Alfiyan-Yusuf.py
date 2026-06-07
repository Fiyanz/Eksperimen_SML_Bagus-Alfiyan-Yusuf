import os
import glob
import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

CLASS_NAMES = ['MP (kutu daun)', 'BT (kutu kebul)', 'T (thrips)', 'C (ulat)']
NUM_CLASSES = len(CLASS_NAMES)

CLASS_MAP = {
    'kutu-daun': 0,
    'kutu-kebul': 1,
    'thrips': 2,
    'thrips-baru': 2,
    'ulat': 3,
}


def get_class_from_filename(filename):
    basename = os.path.basename(filename)
    prefix = basename.split('--')[0].lower()

    if prefix in CLASS_MAP:
        return CLASS_MAP[prefix]

    label_path = filename.rsplit('.', 1)[0] + '.txt'
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                return int(first_line.split()[0])

    raise ValueError(f"Tidak bisa menentukan kelas untuk: {filename}")


def collect_images_and_labels(image_dir):
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    labels = []
    valid_files = []
    for img_path in image_files:
        try:
            label = get_class_from_filename(img_path)
            labels.append(label)
            valid_files.append(img_path)
        except ValueError as e:
            print(f"Peringatan: {e}")

    return valid_files, labels


def load_data(dataset_dir):
    print("Loading training data...")
    train_files, train_labels = collect_images_and_labels(
        os.path.join(dataset_dir, 'train', 'images')
    )
    print(f"  Training: {len(train_files)} images")

    print("Loading validation data...")
    val_files_1, val_labels_1 = collect_images_and_labels(
        os.path.join(dataset_dir, 'val', 'images')
    )
    val_files_2, val_labels_2 = collect_images_and_labels(
        os.path.join(dataset_dir, 'val', 'valid', 'images')
    )
    val_files = val_files_1 + val_files_2
    val_labels = val_labels_1 + val_labels_2
    print(f"  Validation: {len(val_files)} images")

    print("Loading test data...")
    test_files, test_labels = collect_images_and_labels(
        os.path.join(dataset_dir, 'test', 'images')
    )
    print(f"  Test: {len(test_files)} images")

    for split_name, labels in [('Train', train_labels), ('Val', val_labels), ('Test', test_labels)]:
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n  Distribution {split_name}:")
        for cls_id, count in zip(unique, counts):
            print(f"    {CLASS_NAMES[cls_id]}: {count} ({count/len(labels)*100:.1f}%)")

    return train_files, train_labels, val_files, val_labels, test_files, test_labels


def preprocess(train_files, train_labels, val_files, val_labels, test_files, test_labels):
    all_train_files = train_files + val_files
    all_train_labels = train_labels + val_labels

    train_files_final, val_files_final, train_labels_final, val_labels_final = train_test_split(
        all_train_files,
        all_train_labels,
        test_size=0.15,
        stratify=all_train_labels,
        random_state=42
    )

    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.array([0, 1, 2, 3]),
        y=np.array(train_labels_final)
    )
    class_weight_dict = dict(enumerate(class_weights))

    print(f"\nFinal split:")
    print(f"  Train: {len(train_files_final)} images")
    print(f"  Val:   {len(val_files_final)} images")
    print(f"  Test:  {len(test_files)} images")
    print(f"\nClass weights:")
    for i, w in class_weight_dict.items():
        print(f"  {CLASS_NAMES[i]}: weight={w:.4f}")

    return (train_files_final, train_labels_final,
            val_files_final, val_labels_final,
            test_files, test_labels, class_weight_dict)


def build_augmentation():
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.2),
    ], name="data_augmentation")


def parse_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    label = tf.one_hot(label, depth=NUM_CLASSES)
    return img, label


def augment_image(image, label, augmentation):
    image = augmentation(image)
    return image, label


def create_dataset(file_paths, labels, batch_size=BATCH_SIZE, shuffle=False, augment=False, augmentation=None):
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_paths), seed=42)
    ds = ds.map(parse_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    if augment and augmentation is not None:
        ds = ds.map(lambda x, y: augment_image(x, y, augmentation), num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def save_preprocessed(output_dir, train_files, train_labels, val_files, val_labels,
                      test_files, test_labels, class_weight_dict):
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'train_files.npy'), np.array(train_files, dtype=object))
    np.save(os.path.join(output_dir, 'train_labels.npy'), np.array(train_labels))
    np.save(os.path.join(output_dir, 'val_files.npy'), np.array(val_files, dtype=object))
    np.save(os.path.join(output_dir, 'val_labels.npy'), np.array(val_labels))
    np.save(os.path.join(output_dir, 'test_files.npy'), np.array(test_files, dtype=object))
    np.save(os.path.join(output_dir, 'test_labels.npy'), np.array(test_labels))
    np.save(os.path.join(output_dir, 'class_weights.npy'), class_weight_dict)

    metadata = {
        'image_size': IMAGE_SIZE,
        'batch_size': BATCH_SIZE,
        'num_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES,
        'num_train': len(train_files),
        'num_val': len(val_files),
        'num_test': len(test_files),
    }
    np.save(os.path.join(output_dir, 'metadata.npy'), metadata)

    print(f"\nPreprocessed data saved to: {output_dir}")
    print(f"  train_files.npy: {len(train_files)} entries")
    print(f"  val_files.npy:   {len(val_files)} entries")
    print(f"  test_files.npy:  {len(test_files)} entries")
    print(f"  class_weights.npy: {class_weight_dict}")
    print(f"  metadata.npy: image_size={IMAGE_SIZE}, batch_size={BATCH_SIZE}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess Red Chili Pepper Pests Dataset')
    parser.add_argument('--dataset-dir', type=str,
                        default=os.environ.get('DATASET_DIR',
                                               '/kaggle/input/datasets/indraagustian/red-chili-pepper-pests-dataset'),
                        help='Path to raw dataset directory')
    parser.add_argument('--output-dir', type=str,
                        default='red_chili_pepper_pests_preprocessed',
                        help='Path to save preprocessed data')
    args = parser.parse_args()

    print("=" * 60)
    print("Red Chili Pepper Pests - Data Preprocessing Pipeline")
    print("=" * 60)

    train_files, train_labels, val_files, val_labels, test_files, test_labels = load_data(args.dataset_dir)

    (train_files_final, train_labels_final,
     val_files_final, val_labels_final,
     test_files, test_labels, class_weight_dict) = preprocess(
        train_files, train_labels, val_files, val_labels, test_files, test_labels
    )

    augmentation = build_augmentation()
    train_ds = create_dataset(train_files_final, train_labels_final, shuffle=True, augment=True, augmentation=augmentation)
    val_ds = create_dataset(val_files_final, val_labels_final, shuffle=False, augment=False)
    test_ds = create_dataset(test_files, test_labels, shuffle=False, augment=False)

    print(f"\nDataset ready:")
    print(f"  Train batches: {len(list(train_ds))}")
    print(f"  Val batches:   {len(list(val_ds))}")
    print(f"  Test batches:  {len(list(test_ds))}")

    save_preprocessed(args.output_dir,
                      train_files_final, train_labels_final,
                      val_files_final, val_labels_final,
                      test_files, test_labels, class_weight_dict)

    print("\nPreprocessing complete!")


if __name__ == '__main__':
    main()
