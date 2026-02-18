# reference: https://github.com/comfyanonymous/ComfyUI/blob/v0.3.64/comfy/sd.py#L273

import itertools
import math

import torch

from backend import memory_management
from backend.patcher.base import ModelPatcher


@torch.inference_mode()
def tiled_scale_multidim(samples, function, tile=(64, 64), overlap=8, upscale_amount=4, out_channels=3, output_device="cpu", downscale=False, index_formulas=None):
    """https://github.com/comfyanonymous/ComfyUI/blob/v0.3.64/comfy/utils.py#L901"""
    dims = len(tile)

    if not (isinstance(upscale_amount, (tuple, list))):
        upscale_amount = [upscale_amount] * dims

    if not (isinstance(overlap, (tuple, list))):
        overlap = [overlap] * dims

    if index_formulas is None:
        index_formulas = upscale_amount

    if not (isinstance(index_formulas, (tuple, list))):
        index_formulas = [index_formulas] * dims

    def get_upscale(dim, val):
        up = upscale_amount[dim]
        if callable(up):
            return up(val)
        else:
            return up * val

    def get_downscale(dim, val):
        up = upscale_amount[dim]
        if callable(up):
            return up(val)
        else:
            return val / up

    def get_upscale_pos(dim, val):
        up = index_formulas[dim]
        if callable(up):
            return up(val)
        else:
            return up * val

    def get_downscale_pos(dim, val):
        up = index_formulas[dim]
        if callable(up):
            return up(val)
        else:
            return val / up

    if downscale:
        get_scale = get_downscale
        get_pos = get_downscale_pos
    else:
        get_scale = get_upscale
        get_pos = get_upscale_pos

    def mult_list_upscale(a):
        out = []
        for i in range(len(a)):
            out.append(round(get_scale(i, a[i])))
        return out

    output = torch.empty([samples.shape[0], out_channels] + mult_list_upscale(samples.shape[2:]), device=output_device)

    for b in range(samples.shape[0]):
        s = samples[b : b + 1]

        if all(s.shape[d + 2] <= tile[d] for d in range(dims)):
            output[b : b + 1] = function(s).to(output_device)
            continue

        out = torch.zeros([s.shape[0], out_channels] + mult_list_upscale(s.shape[2:]), device=output_device)
        out_div = torch.zeros([s.shape[0], out_channels] + mult_list_upscale(s.shape[2:]), device=output_device)

        positions = [range(0, s.shape[d + 2] - overlap[d], tile[d] - overlap[d]) if s.shape[d + 2] > tile[d] else [0] for d in range(dims)]

        for it in itertools.product(*positions):
            s_in = s
            upscaled = []

            for d in range(dims):
                pos = max(0, min(s.shape[d + 2] - overlap[d], it[d]))
                l = min(tile[d], s.shape[d + 2] - pos)
                s_in = s_in.narrow(d + 2, pos, l)
                upscaled.append(round(get_pos(d, pos)))

            ps = function(s_in).to(output_device)
            mask = torch.ones_like(ps)

            for d in range(2, dims + 2):
                feather = round(get_scale(d - 2, overlap[d - 2]))
                if feather >= mask.shape[d]:
                    continue
                for t in range(feather):
                    a = (t + 1) / feather
                    mask.narrow(d, t, 1).mul_(a)
                    mask.narrow(d, mask.shape[d] - 1 - t, 1).mul_(a)

            o = out
            o_d = out_div
            for d in range(dims):
                o = o.narrow(d + 2, upscaled[d], mask.shape[d + 2])
                o_d = o_d.narrow(d + 2, upscaled[d], mask.shape[d + 2])

            o.add_(ps * mask)
            o_d.add_(mask)

        output[b : b + 1] = out / out_div
    return output


def tiled_scale(samples, function, tile_x=64, tile_y=64, overlap=8, upscale_amount=4, out_channels=3, output_device="cpu"):
    return tiled_scale_multidim(samples, function, (tile_y, tile_x), overlap=overlap, upscale_amount=upscale_amount, out_channels=out_channels, output_device=output_device)


class VAE:
    # Conservative defaults for 12GB VRAM - 512px tiles
    DEFAULT_TILE_SIZE_LATENT = 64   # 64 latent = 512px for 8x compression
    DEFAULT_OVERLAP_LATENT = 8      # 8 latent = 64px overlap

    def __init__(self, model=None, device=None, dtype=None, no_init=False, *, is_wan=False, is_flux2=False):
        if no_init:
            return

        if not is_wan:
            self.upscale_ratio = 8
            self.upscale_index_formula = None
            self.downscale_ratio = 8
            self.downscale_index_formula = None
            self.latent_dim = 2
            self.latent_channels = int(model.config.latent_channels)  # 4 | 16
            self.memory_used_encode = lambda shape, dtype: (1767 * shape[2] * shape[3]) * memory_management.dtype_size(dtype)
            self.memory_used_decode = lambda shape, dtype: (2178 * shape[2] * shape[3] * 64) * memory_management.dtype_size(dtype)

            if is_flux2:
                self.upscale_ratio = 16
                self.downscale_ratio = 16
                self.latent_channels = 128
                self.memory_used_decode = lambda shape, dtype: (2178 * shape[2] * shape[3] * 64) * memory_management.dtype_size(dtype) * 4.0

        else:
            self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 8, 8)
            self.upscale_index_formula = (4, 8, 8)
            self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 8, 8)
            self.downscale_index_formula = (4, 8, 8)
            self.latent_dim = 3
            self.latent_channels = int(model.config.z_dim)  # 16
            self.memory_used_encode = lambda shape, dtype: (1500 if shape[2] <= 4 else 6000) * shape[3] * shape[4] * memory_management.dtype_size(dtype)
            self.memory_used_decode = lambda shape, dtype: (2200 if shape[2] <= 4 else 7000) * shape[3] * shape[4] * (8 * 8) * memory_management.dtype_size(dtype)

        self.output_channels = 3
        self.first_stage_model = model.eval()

        self.device = device or memory_management.vae_device()
        offload_device = memory_management.vae_offload_device()

        self.vae_dtype = dtype or memory_management.vae_dtype()
        self.first_stage_model.to(self.vae_dtype)
        self.output_device = memory_management.intermediate_device()

        self.patcher = ModelPatcher(self.first_stage_model, load_device=self.device, offload_device=offload_device)
        self.is_wan = is_wan

    def clone(self):
        n = VAE(no_init=True)
        n.patcher = self.patcher.clone()
        n.memory_used_encode = self.memory_used_encode
        n.memory_used_decode = self.memory_used_decode
        n.downscale_ratio = self.downscale_ratio
        n.upscale_ratio = self.upscale_ratio
        n.downscale_index_formula = self.downscale_index_formula
        n.upscale_index_formula = self.upscale_index_formula
        n.latent_channels = self.latent_channels
        n.latent_dim = self.latent_dim
        n.output_channels = self.output_channels
        n.first_stage_model = self.first_stage_model
        n.device = self.device
        n.vae_dtype = self.vae_dtype
        n.output_device = self.output_device
        n.is_wan = self.is_wan
        return n

    # ==================== TILING HELPER METHODS ====================

    def _get_spatial_ratio(self):
        """Get the spatial compression ratio for 2D operations."""
        if isinstance(self.upscale_ratio, tuple):
            return self.upscale_ratio[-1]
        return self.upscale_ratio

    @staticmethod
    def _make_linear_ramp(size, direction='up'):
        """Create a 1D linear ramp on CPU for blending."""
        if direction == 'up':
            return torch.linspace(0.0, 1.0, size, dtype=torch.float32)
        else:
            return torch.linspace(1.0, 0.0, size, dtype=torch.float32)

    def _decode_tile_to_cpu(self, samples):
        """Decode a single tile on GPU, return result on CPU immediately."""
        samples_gpu = samples.to(self.vae_dtype).to(self.device)
        decoded = self.first_stage_model.decode(samples_gpu)
        return self.process_output(decoded.float()).cpu()

    def _encode_tile_to_cpu(self, pixels):
        """Encode a single tile on GPU, return result on CPU immediately."""
        pixels_gpu = self.process_input(pixels).to(self.vae_dtype).to(self.device)
        encoded = self.first_stage_model.encode(pixels_gpu)
        return encoded.float().cpu()

    def _decode_tiled_cpu_accumulate(self, samples, tile_x, tile_y, overlap):
        """
        Tiled decode with CPU accumulation and linear blend ramps.
        GPU only processes one tile at a time, all blending on CPU.
        Handles non-square images correctly via independent x/y tile loops.
        """
        b, c, h, w = samples.shape
        ratio = self._get_spatial_ratio()
        out_h = h * ratio
        out_w = w * ratio
        overlap_px = overlap * ratio

        # All accumulation on CPU (uses system RAM)
        output = torch.zeros((b, self.output_channels, out_h, out_w), dtype=torch.float32)
        weights = torch.zeros((1, 1, out_h, out_w), dtype=torch.float32)

        step_x = max(1, tile_x - overlap)
        step_y = max(1, tile_y - overlap)

        # Build tile list - handles non-square images with independent x/y loops
        tiles = []
        y = 0
        while y < h:
            x = 0
            while x < w:
                x_end = min(x + tile_x, w)
                y_end = min(y + tile_y, h)
                tiles.append((x, y, x_end, y_end))
                x += step_x
            y += step_y

        for (x, y, x_end, y_end) in tiles:
            tile = samples[:, :, y:y_end, x:x_end]
            decoded = self._decode_tile_to_cpu(tile)

            ox = x * ratio
            oy = y * ratio
            ox_end = x_end * ratio
            oy_end = y_end * ratio
            th = oy_end - oy
            tw = ox_end - ox

            # Build blend mask on CPU
            blend = torch.ones((1, 1, th, tw), dtype=torch.float32)

            # Left edge blend
            if x > 0 and overlap_px > 0:
                ramp_len = min(overlap_px, tw)
                blend[:, :, :, :ramp_len] *= self._make_linear_ramp(ramp_len, 'up').view(1, 1, 1, -1)

            # Right edge blend
            if x_end < w and overlap_px > 0:
                ramp_len = min(overlap_px, tw)
                blend[:, :, :, -ramp_len:] *= self._make_linear_ramp(ramp_len, 'down').view(1, 1, 1, -1)

            # Top edge blend
            if y > 0 and overlap_px > 0:
                ramp_len = min(overlap_px, th)
                blend[:, :, :ramp_len, :] *= self._make_linear_ramp(ramp_len, 'up').view(1, 1, -1, 1)

            # Bottom edge blend
            if y_end < h and overlap_px > 0:
                ramp_len = min(overlap_px, th)
                blend[:, :, -ramp_len:, :] *= self._make_linear_ramp(ramp_len, 'down').view(1, 1, -1, 1)

            # Accumulate on CPU
            output[:, :, oy:oy_end, ox:ox_end] += decoded * blend
            weights[:, :, oy:oy_end, ox:ox_end] += blend

        return output / weights.clamp(min=1e-8)

    def _encode_tiled_cpu_accumulate(self, pixel_samples, tile_x, tile_y, overlap):
        """
        Tiled encode with CPU accumulation and linear blend ramps.
        GPU only processes one tile at a time, all blending on CPU.
        Handles non-square images correctly via independent x/y tile loops.
        """
        b, c, h, w = pixel_samples.shape
        ratio = self._get_spatial_ratio()
        out_h = h // ratio
        out_w = w // ratio
        overlap_latent = overlap // ratio

        output = torch.zeros((b, self.latent_channels, out_h, out_w), dtype=torch.float32)
        weights = torch.zeros((1, 1, out_h, out_w), dtype=torch.float32)

        step_x = max(ratio, tile_x - overlap)
        step_y = max(ratio, tile_y - overlap)

        tiles = []
        y = 0
        while y < h:
            x = 0
            while x < w:
                x_end = min(x + tile_x, w)
                y_end = min(y + tile_y, h)
                tiles.append((x, y, x_end, y_end))
                x += step_x
            y += step_y

        for (x, y, x_end, y_end) in tiles:
            tile = pixel_samples[:, :, y:y_end, x:x_end]
            encoded = self._encode_tile_to_cpu(tile)

            ox = x // ratio
            oy = y // ratio
            ox_end = x_end // ratio
            oy_end = y_end // ratio
            th = oy_end - oy
            tw = ox_end - ox

            blend = torch.ones((1, 1, th, tw), dtype=torch.float32)

            if x > 0 and overlap_latent > 0:
                ramp_len = min(overlap_latent, tw)
                blend[:, :, :, :ramp_len] *= self._make_linear_ramp(ramp_len, 'up').view(1, 1, 1, -1)

            if x_end < w and overlap_latent > 0:
                ramp_len = min(overlap_latent, tw)
                blend[:, :, :, -ramp_len:] *= self._make_linear_ramp(ramp_len, 'down').view(1, 1, 1, -1)

            if y > 0 and overlap_latent > 0:
                ramp_len = min(overlap_latent, th)
                blend[:, :, :ramp_len, :] *= self._make_linear_ramp(ramp_len, 'up').view(1, 1, -1, 1)

            if y_end < h and overlap_latent > 0:
                ramp_len = min(overlap_latent, th)
                blend[:, :, -ramp_len:, :] *= self._make_linear_ramp(ramp_len, 'down').view(1, 1, -1, 1)

            output[:, :, oy:oy_end, ox:ox_end] += encoded * blend
            weights[:, :, oy:oy_end, ox:ox_end] += blend

        return output / weights.clamp(min=1e-8)

    # ==================== TILED DECODE/ENCODE ====================

    @torch.inference_mode()
    def decode_tiled_(self, samples, tile_x=None, tile_y=None, overlap=None):
        """
        Tiled VAE decode with CPU accumulation.
        One tile at a time on GPU, all blending on CPU.
        """
        b, c, h, w = samples.shape

        if tile_x is None:
            tile_x = self.DEFAULT_TILE_SIZE_LATENT
        if tile_y is None:
            tile_y = self.DEFAULT_TILE_SIZE_LATENT
        if overlap is None:
            overlap = self.DEFAULT_OVERLAP_LATENT

        # Clamp to image size
        tile_x = min(tile_x, w)
        tile_y = min(tile_y, h)

        # Single tile case - no tiling needed
        if w <= tile_x and h <= tile_y:
            return self._decode_tile_to_cpu(samples).to(self.output_device)

        # Clear GPU memory once before starting
        if self.device.type != 'cpu':
            torch.cuda.empty_cache()

        return self._decode_tiled_cpu_accumulate(samples, tile_x, tile_y, overlap).to(self.output_device)

    def decode_tiled_3d(self, samples, tile_t=999, tile_x=32, tile_y=32, overlap=(1, 8, 8)):
        decode_fn = lambda a: self.first_stage_model.decode(a.to(self.vae_dtype).to(self.device)).float()
        return self.process_output(tiled_scale_multidim(samples, decode_fn, tile=(tile_t, tile_x, tile_y), overlap=overlap, upscale_amount=self.upscale_ratio, out_channels=self.output_channels, index_formulas=self.upscale_index_formula, output_device=self.output_device))

    @torch.inference_mode()
    def encode_tiled_(self, pixel_samples, tile_x=None, tile_y=None, overlap=None):
        """
        Tiled VAE encode with CPU accumulation.
        One tile at a time on GPU, all blending on CPU.
        """
        b, c, h, w = pixel_samples.shape
        ratio = self._get_spatial_ratio()

        # Defaults in pixel space
        if tile_x is None:
            tile_x = self.DEFAULT_TILE_SIZE_LATENT * ratio
        if tile_y is None:
            tile_y = self.DEFAULT_TILE_SIZE_LATENT * ratio
        if overlap is None:
            overlap = self.DEFAULT_OVERLAP_LATENT * ratio

        tile_x = min(tile_x, w)
        tile_y = min(tile_y, h)

        if w <= tile_x and h <= tile_y:
            return self._encode_tile_to_cpu(pixel_samples).to(self.output_device)

        if self.device.type != 'cpu':
            torch.cuda.empty_cache()

        return self._encode_tiled_cpu_accumulate(pixel_samples, tile_x, tile_y, overlap).to(self.output_device)

    def encode_tiled_3d(self, samples, tile_t=9999, tile_x=512, tile_y=512, overlap=(1, 64, 64)):
        encode_fn = lambda a: self.first_stage_model.encode((self.process_input(a)).to(self.vae_dtype).to(self.device)).float()
        return tiled_scale_multidim(samples, encode_fn, tile=(tile_t, tile_x, tile_y), overlap=overlap, upscale_amount=self.downscale_ratio, out_channels=self.latent_channels, downscale=True, index_formulas=self.downscale_index_formula, output_device=self.output_device)

    # ==================== PUBLIC API ====================

    def decode(self, samples_in: torch.Tensor):
        """Always uses tiled decode for memory safety."""
        return self.decode_tiled(samples_in)

    def decode_tiled(self, samples: torch.Tensor, tile_x: int = None, tile_y: int = None, overlap: int = None):
        memory_used = self.memory_used_decode(samples.shape, self.vae_dtype)
        memory_management.load_models_gpu([self.patcher], memory_required=memory_used)

        if tile_x is None:
            tile_x = self.DEFAULT_TILE_SIZE_LATENT
        if tile_y is None:
            tile_y = self.DEFAULT_TILE_SIZE_LATENT
        if overlap is None:
            overlap = self.DEFAULT_OVERLAP_LATENT

        if not self.is_wan:
            output = self.decode_tiled_(samples, tile_x=tile_x, tile_y=tile_y, overlap=overlap)
        else:
            output = self.decode_tiled_3d(samples, tile_x=tile_x, tile_y=tile_y, overlap=(1, overlap, overlap))

        return output.movedim(1, -1)

    def encode(self, pixel_samples: torch.Tensor):
        """Always uses tiled encode for memory safety."""
        return self.encode_tiled(pixel_samples)

    def encode_tiled(self, pixel_samples: torch.Tensor, tile_x: int = None, tile_y: int = None, overlap: int = None):
        pixel_samples = pixel_samples.movedim(-1, 1)
        if self.is_wan and pixel_samples.ndim < 5:
            pixel_samples = pixel_samples.movedim(1, 0).unsqueeze(0)

        memory_used = self.memory_used_encode(pixel_samples.shape, self.vae_dtype)
        memory_management.load_models_gpu([self.patcher], memory_required=memory_used)

        ratio = self._get_spatial_ratio()
        if tile_x is None:
            tile_x = self.DEFAULT_TILE_SIZE_LATENT * ratio
        if tile_y is None:
            tile_y = self.DEFAULT_TILE_SIZE_LATENT * ratio
        if overlap is None:
            overlap = self.DEFAULT_OVERLAP_LATENT * ratio

        if not self.is_wan:
            return self.encode_tiled_(pixel_samples, tile_x=tile_x, tile_y=tile_y, overlap=overlap)

        # WAN 3D encoding
        tile_t = self.upscale_ratio[0](9999)
        maximum = self.upscale_ratio[0](self.downscale_ratio[0](pixel_samples.shape[2]))
        return self.encode_tiled_3d(pixel_samples[:, :, :maximum], tile_t=tile_t, tile_x=tile_x, tile_y=tile_y, overlap=(1, overlap, overlap))

    def process_input(self, image):
        return image * 2.0 - 1.0

    def process_output(self, image):
        return torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
