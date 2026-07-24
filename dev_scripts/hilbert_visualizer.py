import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch

from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# CLAS12 detector radii (raw values)
# ============================================================

CLAS12_LAYER_RADII_RAW = [
    6.52944,
    9.28923,
    12.03261,
    14.76460,
    19.26460,
    22.26460,
]

# ============================================================
# Visualization settings
# ============================================================

NUM_BITS = 3               # 8x8 Hilbert grid
GRID = 2 ** NUM_BITS

ETA_MIN = -3.0
ETA_MAX = 3.0

CYLINDER_ALPHA = 0.08
PATH_WIDTH = 2.2

FIGSIZE = (11,10)

# ============================================================
# Hilbert helper functions
# (remaining ones appear in Part 2)
# ============================================================

def right_shift(binary, k=1, axis=-1):

    if binary.shape[axis] <= k:
        return torch.zeros_like(binary)

    slicing = [slice(None)] * len(binary.shape)
    slicing[axis] = slice(None, -k)

    shifted = torch.nn.functional.pad(
        binary[tuple(slicing)],
        (k,0),
        mode="constant",
        value=0
    )

    return shifted


def binary2gray(binary, axis=-1):

    shifted = right_shift(binary, axis=axis)
    return torch.logical_xor(binary, shifted)


def gray2binary(gray, axis=-1):

    shift = 2 ** (
        torch.Tensor([gray.shape[axis]])
        .log2()
        .ceil()
        .int() - 1
    )

    while shift > 0:
        gray = torch.logical_xor(
            gray,
            right_shift(gray, shift)
        )

        shift = torch.div(
            shift,
            2,
            rounding_mode="floor"
        )

    return gray

# ============================================================
# Hilbert decode (exact implementation)
# ============================================================

def decode(hilberts, num_dims, num_bits):

    if num_dims * num_bits > 64:
        raise ValueError

    hilberts = torch.atleast_1d(hilberts)

    orig_shape = hilberts.shape

    bitpack_mask = 2 ** torch.arange(0,8)
    bitpack_mask_rev = bitpack_mask.flip(-1)

    hh_uint8 = (
        hilberts.ravel()
        .type(torch.int64)
        .view(torch.uint8)
        .reshape((-1,8))
        .flip(-1)
    )

    hh_bits = (
        hh_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0)
        .byte()
        .flatten(-2,-1)[:,-num_dims*num_bits:]
    )

    gray = binary2gray(hh_bits)

    gray = (
        gray
        .reshape((-1,num_bits,num_dims))
        .swapaxes(1,2)
    )

    for bit in range(num_bits-1,-1,-1):

        for dim in range(num_dims-1,-1,-1):

            mask = gray[:,dim,bit]

            gray[:,0,bit+1:] = torch.logical_xor(
                gray[:,0,bit+1:],
                mask[:,None]
            )

            to_flip = torch.logical_and(
                torch.logical_not(mask[:,None]),
                torch.logical_xor(
                    gray[:,0,bit+1:],
                    gray[:,dim,bit+1:]
                )
            )

            gray[:,dim,bit+1:] = torch.logical_xor(
                gray[:,dim,bit+1:],
                to_flip
            )

            gray[:,0,bit+1:] = torch.logical_xor(
                gray[:,0,bit+1:],
                to_flip
            )

    extra_dims = 64 - num_bits

    padded = torch.nn.functional.pad(
        gray,
        (extra_dims,0),
        "constant",
        0
    )

    locs_chopped = (
        padded
        .flip(-1)
        .reshape((-1,num_dims,8,8))
    )

    locs_uint8 = (
        (locs_chopped * bitpack_mask)
        .sum(3)
        .squeeze()
        .type(torch.uint8)
    )

    flat_locs = locs_uint8.view(torch.int64)

    return flat_locs.reshape((*orig_shape,num_dims))


# ============================================================
# Build one 2D Hilbert traversal
# ============================================================

def build_hilbert_grid(num_bits):

    N = 2**num_bits

    distances = torch.arange(N*N)

    coords = decode(
        distances,
        num_dims=2,
        num_bits=num_bits
    ).numpy()

    return coords

# ============================================================
# Convert Hilbert grid -> cylindrical coordinates
# ============================================================

coords = build_hilbert_grid(NUM_BITS)

curve_xyz = []

for layer, r in enumerate(CLAS12_LAYER_RADII_RAW):

    xyz = []

    for gx, gy in coords:

        # phi wraps around full cylinder
        phi = 2*np.pi*(gx/(GRID-1))

        # schematic eta axis
        eta = ETA_MIN + (ETA_MAX-ETA_MIN)*(gy/(GRID-1))

        x = r*np.cos(phi)
        y = r*np.sin(phi)

        # linear z for visualization
        z = eta * 4.5

        xyz.append([x,y,z])

    xyz = np.asarray(xyz)

    # append entire Hilbert traversal on this layer
    curve_xyz.extend(xyz)

    # connect to next detector layer
    if layer < len(CLAS12_LAYER_RADII_RAW)-1:

        nxt = CLAS12_LAYER_RADII_RAW[layer+1]

        last = xyz[-1]

        phi_last = np.arctan2(last[1], last[0])
        z_last = last[2]

        curve_xyz.append([
            nxt*np.cos(phi_last),
            nxt*np.sin(phi_last),
            z_last
        ])

curve_xyz = np.asarray(curve_xyz)

# ============================================================
# Plot
# ============================================================

fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_subplot(111, projection="3d")

# transparent cylinders
theta = np.linspace(0,2*np.pi,200)
zline = np.linspace(ETA_MIN*4.5, ETA_MAX*4.5,40)

Theta,Z = np.meshgrid(theta,zline)

for r in CLAS12_LAYER_RADII_RAW:

    X = r*np.cos(Theta)
    Y = r*np.sin(Theta)

    ax.plot_surface(
        X,
        Y,
        Z,
        alpha=CYLINDER_ALPHA,
        linewidth=0,
        color="lightgray",
        shade=False
    )

# color progression
cmap = mpl.cm.viridis

for i in range(len(curve_xyz)-1):

    frac = i/(len(curve_xyz)-2)

    ax.plot(
        curve_xyz[i:i+2,0],
        curve_xyz[i:i+2,1],
        curve_xyz[i:i+2,2],
        color=cmap(frac),
        lw=PATH_WIDTH
    )

# start / end markers
ax.scatter(
    curve_xyz[0,0],
    curve_xyz[0,1],
    curve_xyz[0,2],
    s=120,
    color="red",
    edgecolors="k",
    label="Start"
)

ax.scatter(
    curve_xyz[-1,0],
    curve_xyz[-1,1],
    curve_xyz[-1,2],
    s=120,
    color="gold",
    edgecolors="k",
    label="End"
)

# cosmetics
ax.set_box_aspect((1,1,1.2))

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel(r"$\eta$")

ax.set_title(
    "Modified CLAS12 Hilbert Ordering\n"
    "2D Hilbert traversal on each detector layer\n"
    "Layers traversed sequentially in radius"
)

ax.view_init(elev=22, azim=-55)

# remove panes
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.fill = False
    axis.pane.set_edgecolor("white")

ax.grid(False)

plt.tight_layout()
plt.savefig(
    "modified_hilbert_clas12.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()