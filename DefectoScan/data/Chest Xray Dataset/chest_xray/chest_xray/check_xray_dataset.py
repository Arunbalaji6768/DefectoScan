import os

base_dir = os.path.abspath(os.path.dirname(__file__))
train_dir = os.path.join(base_dir, 'DefectoScan', 'data', 'Chest Xray Dataset', 'chest_xray', 'chest_xray', 'train')
test_dir  = os.path.join(base_dir, 'DefectoScan', 'data', 'Chest Xray Dataset', 'chest_xray', 'chest_xray', 'test')

def count_images(folder):
    if not os.path.exists(folder):
        return 0
    count = 0
    for root, dirs, files in os.walk(folder):
        count += len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    return count

print(f"Checking dataset folders...")

if not os.path.exists(train_dir):
    print(f" Train folder missing: {train_dir}")
else:
    print(f" Train folder found: {train_dir}")
    print(f"   Images: {count_images(train_dir)}")

if not os.path.exists(test_dir):
    print(f" Test folder missing: {test_dir}")
else:
    print(f"Test folder found: {test_dir}")
    print(f"   Images: {count_images(test_dir)}") 