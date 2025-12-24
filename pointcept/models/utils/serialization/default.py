import torch
from .z_order import xyz2key as z_order_encode_
from .z_order import key2xyz as z_order_decode_
from .hilbert import encode as hilbert_encode_
from .hilbert import decode as hilbert_decode_
from .bwms import get_bwms_coder


@torch.inference_mode()
def encode(grid_coord, batch=None, depth=16, order="z", coord=None, atom_type=None, atom_types_json_path=None, pre_bwms_order=None):
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans", "bwms", "pre-bwms"}
    if order == "z":
        code = z_order_encode(grid_coord, depth=depth)
    elif order == "z-trans":
        code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        code = hilbert_encode(grid_coord, depth=depth)
    elif order == "hilbert-trans":
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "bwms":
        if coord is None or atom_type is None:
            raise ValueError("BWMS order requires 'coord' and 'atom_type' as input.")
        if atom_types_json_path is None:
            raise ValueError("BWMS order requires 'atom_types_json_path' as input.")
        code = get_bwms_coder(atom_types_json_path).encode(coord, atom_type, batch=batch)
        code = torch.from_numpy(code).to(grid_coord.device)
    elif order == "pre-bwms":
        if pre_bwms_order is None:
            raise ValueError("Serialization order 'pre-bwms' requires pre-computed order as input.")
        code = pre_bwms_order.to(grid_coord.device).long()
    else:
        raise NotImplementedError
    if batch is not None:
        batch = batch.long()
        code = batch << depth * 3 | code
    return code


@torch.inference_mode()
def decode(code, depth=16, order="z"):
    assert order in {"z", "hilbert"}
    batch = code >> depth * 3
    code = code & ((1 << depth * 3) - 1)
    if order == "z":
        grid_coord = z_order_decode(code, depth=depth)
    elif order == "hilbert":
        grid_coord = hilbert_decode(code, depth=depth)
    else:
        raise NotImplementedError
    return grid_coord, batch


def z_order_encode(grid_coord: torch.Tensor, depth: int = 16):
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    # we block the support to batch, maintain batched code in Point class
    code = z_order_encode_(x, y, z, b=None, depth=depth)
    return code


def z_order_decode(code: torch.Tensor, depth):
    x, y, z = z_order_decode_(code, depth=depth)
    grid_coord = torch.stack([x, y, z], dim=-1)  # (N,  3)
    return grid_coord


def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16):
    return hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)


def hilbert_decode(code: torch.Tensor, depth: int = 16):
    return hilbert_decode_(code, num_dims=3, num_bits=depth)
