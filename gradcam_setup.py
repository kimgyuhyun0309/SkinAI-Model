"""
gradcam_setup.py
- EfficientNet 모델에 Grad-CAM용 forward/backward hook 등록
- gradcam_activations / gradcam_gradients 변수를 외부에서 사용할 수 있도록 제공
"""

import torch


# ============================================================
# 전역 변수 (Grad-CAM 텐서 저장)
# ============================================================
gradcam_activations = None
gradcam_gradients = None


# ============================================================
# Hook 함수 정의
# ============================================================
def forward_hook(module, input, output):
    """
    Forward 시 feature map을 저장
    """
    global gradcam_activations
    gradcam_activations = output.detach()


def backward_hook(module, grad_input, grad_output):
    """
    Backward 시 gradient 저장
    """
    global gradcam_gradients
    gradcam_gradients = grad_output[0].detach()


# ============================================================
# Hook 등록 함수
# ============================================================
def register_gradcam_hooks(model):
    """
    EfficientNet 모델의 마지막 convolution block에
    forward/backward hook을 등록한다.
    
    예:
        from gradcam_setup import register_gradcam_hooks
        register_gradcam_hooks(model)
    """
    # EfficientNet 구조 기준: 마지막 블록 = model.blocks[-1]
    target_layer = model.blocks[-1]

    # Forward hook
    target_layer.register_forward_hook(forward_hook)

    # Backward hook
    target_layer.register_full_backward_hook(backward_hook)

    print("✅ Grad-CAM Hook 등록 완료!")
    return target_layer


# ============================================================
# 모듈 테스트 (직접 실행하는 경우)
# ============================================================
if __name__ == "__main__":
    import timm

    model = timm.create_model("tf_efficientnet_b2_ns", pretrained=False, num_classes=9)
    register_gradcam_hooks(model)