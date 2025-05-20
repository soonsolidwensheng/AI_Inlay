
# Define custom function directory
ARG FUNCTION_DIR="/function"
# 使用基础镜像
FROM python:3.11.11 as build-image

# Include global arg in this stage of the build
ARG FUNCTION_DIR
RUN mkdir -p ${FUNCTION_DIR}
COPY . ${FUNCTION_DIR}
# Install the function's dependencies
# RUN pip install \
#     --target ${FUNCTION_DIR} \
#         awslambdaric
# Copy and install dependencies first (利用缓存)
# COPY ./requirements.txt ${FUNCTION_DIR}/requirements.txt
RUN pip install -r ${FUNCTION_DIR}/requirements.txt

# RUN sed -i '284s/.*/    return closest, distance, triangle_id/' /usr/local/lib/python3.8/site-packages/trimesh/proximity.py
# Copy function code

RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main" > /etc/apt/sources.list
RUN echo "deb-src https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main" >> /etc/apt/sources.list

# Install additional packages
RUN apt-get update && apt-get install -y\
    libgl1-mesa-glx libxi6 libxkbcommon0\
    && rm -rf /var/lib/apt/lists/*
COPY ./object_print3d_utils /usr/local/lib/python3.11/site-packages/bpy/4.2/scripts/addons_core/object_print3d_utils

# RUN mv ${FUNCTION_DIR}/${WORK_DIR}/remesh.py /usr/local/lib/python3.8/site-packages/trimesh/remesh.py
# Set working directory to function root directory
WORKDIR ${FUNCTION_DIR}
ENV LD_LIBRARY_PATH=${FUNCTION_DIR}

EXPOSE 9999
CMD [ "python deploy.py" ]
# Set runtime interface client as default command for the container runtime
# ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]

# Pass the name of the function handler as an argument to the runtime
# CMD [ "stdcrown.handler" ]
# CMD [ "stitch_edge.handler" ]
# CMD [ "mesh_repair.handler" ]
# CMD [ "postprocess.handler" ]
# CMD [ "occlusion.handler" ]


