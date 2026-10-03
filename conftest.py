"""测试根路径注入：保证 `import sindyforge` 在本地与 CI 一致可用。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
