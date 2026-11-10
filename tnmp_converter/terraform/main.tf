terraform {
  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
  }
}

provider "digitalocean" {
  token = var.digitalocean_token
}

resource "digitalocean_ssh_key" "public_key" {
  name       = "tnmp_converter_key"
  public_key = var.public_key
}

resource "digitalocean_droplet" "server" {
  name   = "tnmp-converter"
  region = "fra1"
  size   = "s-1vcpu-2gb"
  image  = "ubuntu-24-04-x64"

  ssh_keys = [
    digitalocean_ssh_key.public_key.id
    ]
}


output "droplet_ip" {
  value = digitalocean_droplet.server.ipv4_address
}
