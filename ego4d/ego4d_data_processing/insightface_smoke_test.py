#!/usr/bin/env python3
"""
독립 실행형 InsightFace 무결점 테스트 스크립트
이 스크립트는 InsightFace 설정이 완벽한지 확인합니다.
"""

import os
import sys
import numpy as np


def main():
    print("🧪 InsightFace 무결점 테스트")
    print("=" * 50)
    
    try:
        # Step 1: Import tests
        print("📦 Step 1: Import testing...")
        import onnxruntime as ort
        import insightface
        from insightface.app import FaceAnalysis
        print("✅ All imports successful")
        
        # Step 2: Environment check
        print("\n🔧 Step 2: Environment check...")
        home = os.environ.get("INSIGHTFACE_HOME")
        if not home:
            print("❌ INSIGHTFACE_HOME environment variable not set")
            print("   Please set: export INSIGHTFACE_HOME=/path/to/insightface/cache")
            return False
        
        print(f"   INSIGHTFACE_HOME: {home}")
        
        # Step 3: ONNX Runtime providers
        print("\n🖥️  Step 3: ONNX Runtime providers...")
        providers = ort.get_available_providers()
        print(f"   Available providers: {providers}")
        
        if "CUDAExecutionProvider" not in providers:
            print("⚠️  CUDAExecutionProvider not available - using CPU mode")
            target_provider = "CPUExecutionProvider"
            ctx_id = -1
        else:
            print("✅ CUDAExecutionProvider available")
            target_provider = "CUDAExecutionProvider"
            ctx_id = 0
        
        # Step 4: Model files check
        print("\n📁 Step 4: Model files check...")
        root = os.path.join(home, "models", "antelopev2")
        need = ["scrfd_10g_bnkps.onnx", "glintr100.onnx", "genderage.onnx", "2d106det.onnx"]
        missing = [f for f in need if not os.path.exists(os.path.join(root, f))]
        
        print(f"   Model directory: {root}")
        if missing:
            print(f"❌ Missing model files: {missing}")
            print("   Please download antelopev2 models:")
            print("   python -c \"from insightface.app import FaceAnalysis; FaceAnalysis(name='antelopev2')\"")
            return False
        
        print("✅ All required model files found:")
        for model in need:
            model_path = os.path.join(root, model)
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"     {model}: {size_mb:.1f} MB")
        
        # Step 5: FaceAnalysis initialization
        print(f"\n🚀 Step 5: FaceAnalysis initialization ({target_provider})...")
        app = FaceAnalysis(name="antelopev2", providers=[target_provider])
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        print("✅ FaceAnalysis initialized successfully")
        
        # Step 6: Dummy inference test
        print("\n🎯 Step 6: Dummy inference test...")
        img = np.zeros((480, 640, 3), dtype=np.uint8)  # Black image
        faces = app.get(img)
        print(f"✅ Inference successful - detected {len(faces)} faces (expected: 0)")
        
        # Step 7: Test with random noise (should detect 0 faces)
        print("\n🎲 Step 7: Random noise test...")
        noise_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        faces_noise = app.get(noise_img)
        print(f"✅ Noise test successful - detected {len(faces_noise)} faces")
        
        # Step 8: Memory and resource check
        print("\n💾 Step 8: Memory check...")
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        print(f"✅ Current memory usage: {memory_mb:.1f} MB")
        
        print("\n" + "=" * 50)
        print("🎉 모든 테스트 통과!")
        print("✅ InsightFace가 올바르게 설정되었습니다.")
        print(f"✅ Provider: {target_provider}")
        print(f"✅ Model: antelopev2")
        print(f"✅ Detection size: 640x640")
        print("=" * 50)
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Please install required packages:")
        print("   pip install insightface onnxruntime-gpu")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print("   Traceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
