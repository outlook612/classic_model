classic_model
===========================

细节见:https://www.yuque.com/wei_lai_ke_ji/clamod

# Configure
>Ubuntu 24.04.2 LTS
>
>VS Code + SSH


# built environment 

## clash for linux
-[clash-for-linux](https://github.com/nelvko/clash-for-linux-install/tree/master)


### transformer 
> 安装最基础的python和torch就能跑

```bash
conda create --name transformer python=3.10 -y

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

