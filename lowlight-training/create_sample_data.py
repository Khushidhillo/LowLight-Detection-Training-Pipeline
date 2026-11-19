"""Create a small sample dataset in data/exdark for quick testing without raw download."""
from pathlib import Path
from PIL import Image, ImageDraw
from lowlight.data import write_yaml

def create_sample_dataset(out_dir="data/exdark", num_images=10):
    root = Path(out_dir)
    if root.exists():
        print(f"Dataset directory '{root}' already exists.")
        return
    for si, split in enumerate(['train', 'val', 'test']):
        (root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (root / 'labels' / split).mkdir(parents=True, exist_ok=True)
        for i in range(num_images):
            im = Image.new('RGB', (640, 640), (15 + si*5 + i, 15, 20))
            draw = ImageDraw.Draw(im)
            draw.rectangle((100, 100, 300, 300), fill=(80 + i*5, 60, 40))
            draw.rectangle((350, 200, 500, 450), fill=(50, 70 + i*5, 30))
            im.save(root / 'images' / split / f'sample_{split}_{i}.png')
            # 12 ExDark classes: Bicycle(0), Boat(1), Bottle(2), Bus(3), Car(4), Cat(5), Chair(6), Cup(7), Dog(8), Motorbike(9), People(10), Table(11)
            label_text = f"4 0.3125 0.3125 0.3125 0.3125\n10 0.664 0.507 0.234 0.390\n"
            (root / 'labels' / split / f'sample_{split}_{i}.txt').write_text(label_text)
    write_yaml(root)
    print(f"Sample dataset successfully created at '{root}'!")

if __name__ == '__main__':
    create_sample_dataset()
