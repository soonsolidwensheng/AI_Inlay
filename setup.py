from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import glob
import os


py_files = glob.glob("**/*.py", recursive=True)
# 将文件路径转换为有效的模块名称
def get_module_name(file_path):
    module_name = os.path.splitext(file_path)[0]
    module_name = module_name.replace(os.sep, '.')
    return module_name

extensions = [Extension(get_module_name(f), [f]) for f in py_files]
setup(ext_modules=cythonize(extensions, language_level="3"))