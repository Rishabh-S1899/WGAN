import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as realsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from WGAN import Discriminator, Generator, initialize_weights

# Hyperparameters etc
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 128
NUM_EPOCHS = 10
FEATURES_disc = 64
FEATURES_GEN = 64
disc_ITERATIONS = 2
WEIGHT_CLIP = 0.01

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

#realset = realsets.CIFAR10(root="./data", transform=transforms, download=True)
#c0omment mnist and uncomment below if you want to train on CelebA realset
realset = realsets.ImageFolder(root=r"celeb_dataset", transform=transforms)
loader = DataLoader(realset, batch_size=BATCH_SIZE, shuffle=True)

# initialize gen and disc/disc
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_disc).to(device)
initialize_weights(gen)
initialize_weights(disc)

# initializate optimizer
opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
opt_disc = optim.RMSprop(disc.parameters(), lr=LEARNING_RATE)

# for tensorboard plotting
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
disc.train()

gen_losses = []
disc_losses = []
disc_accuracies = []

for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        # Train disc: max E[disc(real)] - E[disc(fake)]
        for _ in range(disc_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise)
            disc_real = disc(real).reshape(-1)
            disc_fake = disc(fake).reshape(-1)
            loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake))
            disc.zero_grad()
            loss_disc.backward(retain_graph=True)
            opt_disc.step()

            # clip disc weights between -0.01, 0.01
            for p in disc.parameters():
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        # Train Generator: max E[disc(gen_fake)] <-> min -E[disc(gen_fake)]
        gen_fake = disc(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        disc_real_pred = (disc_real > 0).float()
        disc_fake_pred = (disc_fake <= 0).float()
        disc_accuracy = (torch.sum(disc_real_pred) + torch.sum(disc_fake_pred)) / (2 * cur_batch_size)
        gen_losses.append(loss_gen.item())
        disc_losses.append(loss_disc.item())
        disc_accuracies.append(disc_accuracy)
        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0 and batch_idx > 0:
            gen.eval()
            disc.eval()
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} "
                f"Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}, "
                f"D Accuracy: {disc_accuracy:.4f}"
            )

            with torch.no_grad():
                fake = gen(noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            writer_real.add_scalar("Loss/Discriminator", loss_disc, global_step=step)
            writer_real.add_scalar("Loss/Generator", loss_gen, global_step=step)
            writer_real.add_scalar("Accuracy/Discriminator", disc_accuracy, global_step=step)
            
            step += 1
            gen.train()
            disc.train()
