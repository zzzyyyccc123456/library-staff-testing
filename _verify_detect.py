"""检查 ultralytics 版本和 torch 兼容性"""
import os
os.environ["ULTRALYTICS_DISABLE_REQUIREMENTS"] = "1"

log = open("d:/51.4/5555/verify_log.txt", "w")
log.write("Starting\n"); log.flush()

import torch
log.write("torch: %s\n" % torch.__version__); log.flush()

try:
    import ultralytics
    log.write("ultralytics: %s\n" % ultralytics.__version__); log.flush()
    
    # Check ultralytics init file for lazy loading
    init_path = os.path.join(os.path.dirname(ultralytics.__file__), "__init__.py")
    log.write("ultralytics init: %s\n" % init_path); log.flush()
    
    # Try to find where YOLO is defined in the package
    model_init_path = os.path.join(os.path.dirname(ultralytics.__file__), "models", "__init__.py")
    if os.path.exists(model_init_path):
        with open(model_init_path) as f:
            log.write("models/__init__.py:\n%s\n" % f.read()); log.flush()
except Exception as e:
    log.write("ERR: %s\n" % e); log.flush()

log.close()
