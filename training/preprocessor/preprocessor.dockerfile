# Dockerfile for the image preprocessor.

FROM ubuntu

# Prevents questions from hanging the build
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
		apt-get install -y python3-pip && \
		apt-get install -y libglib2.0-0 
        
RUN pip3 install --break-system-packages PyYAML 
RUN pip3 install --break-system-packages numpy
RUN apt-get install -y libsm6 libxext6 libxrender-dev 
RUN pip3 install --break-system-packages opencv-python
		
# ... existing Dockerfile content ...

# Update package list and install OpenGL library
RUN apt-get update && apt-get install -y libgl1

# ... existing Dockerfile content ...
		
ADD preprocessor.py /
ADD preprocessor.yml /
