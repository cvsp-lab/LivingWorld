# kf_update.py
import torch
from einops import rearrange
from kornia.geometry import PinholeCamera

def _make_pixel_grid(H, W, device="cuda"):
    xs, ys = torch.meshgrid(
        torch.arange(W, device=device, dtype=torch.float32),
        torch.arange(H, device=device, dtype=torch.float32),
        indexing="xy"
    )
    return torch.stack([xs.reshape(-1),
                        ys.reshape(-1),
                        torch.ones(H*W, device=device)], dim=-1)  # [(H*W),3]

def convert_pytorch3d_kornia(camera, focal_length, size=512):
    M = camera.get_world_to_view_transform().get_matrix()[0]         # [4,4] (row-major)
    M = M.transpose(0, 1)                                            # pytorch3d → column-major
    pt3d_to_kornia = torch.diag(torch.tensor([-1., -1, 1, 1], device=camera.device))
    extrinsics = (pt3d_to_kornia @ M).unsqueeze(0)                   # [1,4,4]

    h = torch.tensor([size], device=camera.device)
    w = torch.tensor([size], device=camera.device)
    K = torch.eye(4, device=camera.device)[None]
    K[0, 0, 2] = size // 2
    K[0, 1, 2] = size // 2
    K[0, 0, 0] = focal_length
    K[0, 1, 1] = focal_length
    return PinholeCamera(K, extrinsics, h, w)

class Unproject:
    def __init__(self, H=512, W=512, focal=512.0, device="cuda"):
        self.H, self.W = int(H), int(W)
        self.device = device
        self.init_focal_length = float(focal)

        self.image_latest  = None    # [1,3,H,W] float cuda 0~1
        self.depth_latest  = None    # [1,1,H,W] float cuda
        self.current_camera = None   # pytorch3d Cameras

        # [(H*W),3] homogeneous pixel grid [x,y,1]
        self.points = _make_pixel_grid(self.H, self.W, device)

        # pc slots
        self.current_pc = None
        self.current_pc_sky = None
        self.current_pc_layer = None
        self.current_pc_latest = None
        self.current_pc_layer_latest = None

    @staticmethod
    def convert_pytorch3d_kornia(camera, focal_length, size=512):
        return convert_pytorch3d_kornia(camera, focal_length, size)


    @torch.no_grad()
    def update_current_pc_by_kf(self, valid_mask=None, gen_layer=False, image=None, depth=None, camera=None):
        # 인자 없으면 보관된 최신값 사용
        image  = self.image_latest  if image  is None else image
        depth  = self.depth_latest  if depth  is None else depth
        camera = self.current_camera if camera is None else camera

        assert image is not None and depth is not None and camera is not None, "image/depth/camera 필요"

        H, W = image.shape[-2], image.shape[-1]
        kf_cam = convert_pytorch3d_kornia(camera, self.init_focal_length, size=W)

        # depth → [(H*W),1]
        point_depth = rearrange(depth, "b c h w -> (w h b) c")

        # normals (여기선 더미; 네 클래스의 get_normal이 있으면 교체)
        # normals_world = kf_cam.rotation_matrix.inverse() @ rearrange(normals, 'b c h w -> b c (h w)')
        new_normals = None

        # unproject + colors
        new_points_3d = kf_cam.unproject(self.points, point_depth)           # [(H*W),3]
        new_colors    = rearrange(image, "b c h w -> (w h b) c")             # [(H*W),3]

        # valid mask 필터링
        if valid_mask is not None:
            extract_mask = rearrange(valid_mask, "b c h w -> (w h b) c")[:, 0].bool()
            new_points_3d = new_points_3d[extract_mask]
            new_colors    = new_colors[extract_mask]
            if new_normals is not None:
                new_normals = new_normals[extract_mask]

        self.update_current_pc(new_points_3d, new_colors, gen_layer=gen_layer, normals=new_normals)
        return new_points_3d, new_colors