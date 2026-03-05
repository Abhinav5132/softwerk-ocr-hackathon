# Softwerk-Ocr-Hackathon

## To build:
1. Ensure you have nvidia-container-toolkit installed. Refer here for install instructions: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html 

2. Ensure you have docker buildx installed. Refer here for install instructions: https://github.com/docker/buildx?tab=readme-ov-file#installing (Should be pre installed if you have docker desktop)

3. Configure docker to allow GPU pass through using nvidia-container-toolkit:
    ```bash
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker 
    ```
4. Figure out what gpu architecture to target: 
    run the following
    ```bash 
        nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    ```
    The command returns a number eg: 8.6 remove the . and change CUDA_TARGET in the docker compose file to this number.

    Look for the GPU name in the following table, then cross reference:

    | GPU | Compute Cap |
    |---|---|
    | RTX 40 series (4090, 4080 etc) | 89 |
    | RTX 30 series (3090, 3080, 3070 etc) | 86 |
    | RTX 20 series (2080, 2070 etc) | 75 |
    | GTX 16 series (1660 etc) | 75 |
    | GTX 10 series (1080, 1070 etc) | 61 |
    | A100 | 80 |
    | H100 | 90 |

5. Build the container: 
    ```bash 
    docker compose build
    ```
7. Create a new folder at projet root called data. Create two folders inside:
    images, output. 
    you should have the following file structure:
    
    ```
    Softwerk-ocr-hackathon
    ├── .gitignore
    ├── .dockerignore
    ├── cargo.lock
    ├── cargo.toml
    ├── docker-compose-yml
    ├── dockerfile
    ├── README.md
    ├── src
    └── data
        ├── images
        ├── output
        ├── Pdf's to OCR
        └── murderer.pdf
        
    ```

6. Run the container: 
    ```bash 
    docker compose up
    ```

7. Output of the pipeline can be found in the output folder with each page being its own markdown document. Please use a mardown reader like obsidian or VsCode to read this. 

**NOTE** Please clean the images directory before running again. This should happen automatically but some files might remain.