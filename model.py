"""
Backbone + PAN-FPN + Head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, resnet101
from csp_backbone import CSPDarknet


class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, reg_max=16):
        super(DetectionHead, self).__init__()
        # num_anchors is ignored in anchor-free mode, but kept for interface compatibility
        self.num_classes = num_classes
        self.reg_max = reg_max
        
        # Stem 
        self.stem = nn.Conv2d(in_channels, in_channels, 1)
        
        # classification branch
        self.cls_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 1)
        
        # regression branch
        self.reg_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.reg_pred = nn.Conv2d(in_channels, 4 * (reg_max + 1), 1)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            output: [B, H, W, 4*(reg_max+1) + num_classes]
        """
        B = x.size(0)
        x = self.stem(x)

        cls_feat = self.cls_convs(x)
        cls_output = self.cls_pred(cls_feat) # [B, NC, H, W]

        reg_feat = self.reg_convs(x)
        reg_output = self.reg_pred(reg_feat) # [B, 4*(reg_max+1), H, W]

        output = torch.cat([reg_output, cls_output], dim=1) # [B, C_total, H, W]
        
        # Permute: [B, C, H, W] -> [B, H, W, C]
        output = output.permute(0, 2, 3, 1).contiguous()
        
        return output


class FPN(nn.Module):

    def __init__(self, in_channels_list, out_channels=256):
        super(FPN, self).__init__()
        
        # Lateral layers  
        self.lateral_layers = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1)
            for in_channels in in_channels_list
        ])
        
        # Output layers 
        self.output_layers = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])
    
    def forward(self, features):
        """
        Args:
            features: List of feature maps from backbone [C3, C4, C5]
        Returns:
            outputs: List of FPN feature maps [P3, P4, P5]
        """
        # Build top-down
        laterals = [lateral(feature) for lateral, feature in 
                   zip(self.lateral_layers, features)]
        
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], 
                size=laterals[i - 1].shape[2:],
                mode='nearest'
            )
        
        # Output layers
        outputs = [output(lateral) for output, lateral in 
                  zip(self.output_layers, laterals)]
        
        return outputs


class PANFPN(nn.Module):
    """
    PAN-FPN
    Top-Down: FPN （C3, C4, C5 -> P3_td, P4_td, P5_td）
    Bottom-Up: Path Aggregation（P3_td, P4_td, P5_td -> P3, P4, P5）
    """
    def __init__(self, in_channels_list, out_channels=256):
        super(PANFPN, self).__init__()

        # 1. Top-Down 部分（FPN）
        self.lateral_layers = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1)
            for in_channels in in_channels_list
        ])

        self.fpn_output_layers = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])

        # 2. Bottom-Up 部分（PAN）
        num_levels = len(in_channels_list)

        self.pan_downsample_layers = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
            for _ in range(num_levels - 1)
        ])

        self.pan_output_layers = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in range(num_levels)
        ])

    def forward(self, features):
        laterals = [lateral(feat) for lateral, feat in
                    zip(self.lateral_layers, features)]  # [L3, L4, L5]

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],  # 对齐空间尺寸
                mode='nearest'
            )

        fpn_feats = [output(lateral) for output, lateral in
                     zip(self.fpn_output_layers, laterals)]

        pan_feats = [None] * len(fpn_feats)
        pan_feats[0] = self.pan_output_layers[0](fpn_feats[0])
        for i in range(1, len(fpn_feats)):
            down = self.pan_downsample_layers[i - 1](pan_feats[i - 1])  
            fused = fpn_feats[i] + down                                 
            pan_feats[i] = self.pan_output_layers[i](fused)             

        return pan_feats


class ObjectDetector(nn.Module):
    def __init__(self, num_classes, backbone='cspdarknet_s', num_anchors=1):
        super(ObjectDetector, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # backbone
        if backbone == 'resnet18':
            self.backbone = resnet18(pretrained=True)
            fpn_in_channels = [128, 256, 512]
            self._init_resnet_layers()
        elif backbone == 'resnet34':
            self.backbone = resnet34(pretrained=True)
            fpn_in_channels = [128, 256, 512]
            self._init_resnet_layers()
        elif backbone == 'resnet50':
            self.backbone = resnet50(pretrained=True)
            fpn_in_channels = [512, 1024, 2048]
            self._init_resnet_layers()
        elif backbone == 'cspdarknet_n':
            self.backbone = CSPDarknet(depth_multiple=0.33, width_multiple=0.25)
            fpn_in_channels = self.backbone.out_channels
        elif backbone == 'cspdarknet_s':
            self.backbone = CSPDarknet(depth_multiple=0.33, width_multiple=0.50)
            fpn_in_channels = self.backbone.out_channels
        elif backbone == 'cspdarknet_m':
            self.backbone = CSPDarknet(depth_multiple=0.67, width_multiple=0.75)
            fpn_in_channels = self.backbone.out_channels
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.backbone_type = 'resnet' if 'resnet' in backbone else 'csp'

        fpn_out_channels = 256
        self.fpn = PANFPN(fpn_in_channels, fpn_out_channels)
        
        # detection heads
        self.detect_heads = nn.ModuleList([
            DetectionHead(fpn_out_channels, num_anchors, num_classes)
            for _ in range(3)
        ])
        

        self.grids = [None] * 3
        
    def _init_resnet_layers(self):

        self.conv1 = self.backbone.conv1
        self.bn1 = self.backbone.bn1
        self.relu = self.backbone.relu
        self.maxpool = self.backbone.maxpool
        
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            predictions: List of 3 tensors
                Each: [B, num_anchors, Hi, Wi, 5+num_classes]
        """
        if self.backbone_type == 'resnet':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            c2 = self.layer1(x)
            c3 = self.layer2(c2)
            c4 = self.layer3(c3)
            c5 = self.layer4(c4)
            features = [c3, c4, c5]
        else:
            features = self.backbone(x)
        
        fpn_features = self.fpn(features)
        
        predictions = []
        for i, (feature, head) in enumerate(zip(fpn_features, self.detect_heads)):
            pred = head(feature)
            predictions.append(pred)
        
        return predictions
    
    def decode_predictions(self, predictions, conf_threshold=0.5, feature_strides=(8, 16, 32)):
        device = predictions[0].device
        batch_size = predictions[0].size(0)
        all_detections = [[] for _ in range(batch_size)]

        reg_max = 16  # should match head / loss
        project = torch.arange(reg_max + 1, dtype=torch.float, device=device)

        for level, pred in enumerate(predictions):
            # pred: [B, H, W, 4*(reg_max+1) + num_classes]
            B, H, W, _ = pred.shape
            stride = feature_strides[level] if level < len(feature_strides) else feature_strides[-1]

            in_h = H * stride
            in_w = W * stride

            # Split: cls prob + reg dist
            pred_cls = pred[..., -self.num_classes:].sigmoid()  # [B, H, W, NC]
            pred_reg_dist = pred[..., :-self.num_classes]       # [B, H, W, 4*(reg_max+1)]
            pred_reg_dist = pred_reg_dist.view(B, H, W, 4, reg_max + 1).softmax(dim=-1)

            # DFL Expectation in bins: [B, H, W, 4]
            pred_ltrb = torch.matmul(pred_reg_dist, project)
            # Convert bins -> pixels
            pred_ltrb = pred_ltrb * float(stride)

            # Anchor points in pixels: (x+0.5, y+0.5) * stride
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )
            ax = (grid_x.float() + 0.5) * float(stride)
            ay = (grid_y.float() + 0.5) * float(stride)

            for b in range(B):
                scores = pred_cls[b]  # [H, W, NC]
                class_conf, class_id = torch.max(scores, dim=-1)  # [H, W]
                conf_mask = class_conf > conf_threshold
                if conf_mask.sum() == 0:
                    continue

                valid_ltrb = pred_ltrb[b][conf_mask]  # [N, 4] in pixels
                vy, vx = torch.where(conf_mask)

                anc_x = ax[vy, vx]
                anc_y = ay[vy, vx]

                x1 = anc_x - valid_ltrb[:, 0]
                y1 = anc_y - valid_ltrb[:, 1]
                x2 = anc_x + valid_ltrb[:, 2]
                y2 = anc_y + valid_ltrb[:, 3]

                # Clip to image bounds (optional, helps stability)
                x1 = x1.clamp(0, in_w)
                y1 = y1.clamp(0, in_h)
                x2 = x2.clamp(0, in_w)
                y2 = y2.clamp(0, in_h)

                w_box = (x2 - x1).clamp(min=0)
                h_box = (y2 - y1).clamp(min=0)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # Normalize to input size
                cx /= float(in_w)
                cy /= float(in_h)
                w_box /= float(in_w)
                h_box /= float(in_h)

                valid_conf = class_conf[conf_mask]
                valid_cls = class_id[conf_mask].float()

                detections = torch.stack([valid_cls, valid_conf, cx, cy, w_box, h_box], dim=1)
                all_detections[b].append(detections)

        # Merge across levels
        for b in range(batch_size):
            if len(all_detections[b]) > 0:
                all_detections[b] = torch.cat(all_detections[b], dim=0)
            else:
                all_detections[b] = torch.zeros((0, 6), device=device)

        return all_detections


def build_model(config):
    num_classes = config['data']['num_classes']
    backbone = config['model']['backbone']
    num_anchors = config['model']['num_anchors']
    
    model = ObjectDetector(
        num_classes=num_classes,
        backbone=backbone,
        num_anchors=num_anchors
    )
    
    return model


if __name__ == "__main__":
    model = ObjectDetector(num_classes=80, backbone='cspdarknet_s', num_anchors=1)
    print(model)
    # model = ObjectDetectorpanfpn(num_classes=80, backbone='resnet101', num_anchors=1)
    model.eval()
    x = torch.randn(2, 3, 640, 640)
    
    with torch.no_grad():
        predictions = model(x)
        print("Model output:")
        for i, pred in enumerate(predictions):
            print(f"  Level {i}: {pred.shape}")
        detections = model.decode_predictions(predictions, conf_threshold=0.5)
        print(f"\nDetections per image: {[len(d) for d in detections]}")
