# OCI Always Free Deployment (24/7)

This project is a Flask app (`app.py`) that serves:
- `GET /` UI
- `GET /status`
- `POST /predict` upload inference

These steps target **Oracle Cloud Always Free** on an **Ampere A1 (ARM64)** VM running **Ubuntu 22.04 LTS** (recommended so `python3` is 3.10, which is typically easiest for TensorFlow wheels).

## 1) Create the VM (Oracle Console)

1. Create an **Always Free eligible** compute instance:
   - Shape: `VM.Standard.A1.Flex` (ARM)
   - OCPU/Memory: start with `2 OCPU / 12 GB` (increase later if needed)
   - OS: Ubuntu 22.04 LTS
2. Networking:
   - Put it in a public subnet with a public IPv4.
3. Add an ingress rule on the VCN Security List (or Network Security Group) for:
   - TCP `80` (HTTP)

If you prefer serving directly without nginx, open TCP `5050` instead, but nginx on `80` is recommended.

## 2) SSH In

```bash
ssh -i /path/to/key ubuntu@YOUR_PUBLIC_IP
```

## 3) Install + Run

On the VM:

```bash
sudo apt-get update
sudo apt-get install -y git

git clone https://github.com/Srinidhi262005/DeepShield-AI.git
cd DeepShield-AI

sudo bash deploy/oci/install.sh
sudo systemctl enable --now deepshield
sudo systemctl status deepshield --no-pager
```

## 4) Verify

From your laptop:

```bash
curl -s http://YOUR_PUBLIC_IP/status
```

You should see JSON with `"status":"online"`.

## Notes / Costs

- Keeping it Always Free depends on staying within Oracle's Always Free limits for A1 OCPU-hours and GB-hours.
- TensorFlow can be memory heavy. If you see OOM kills, reduce workers (we already run 1 worker) or use a smaller VM config.
