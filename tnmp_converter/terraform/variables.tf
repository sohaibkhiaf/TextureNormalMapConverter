variable "digitalocean_token" {
  description = "DigitalOcean API token"
  type        = string
  sensitive   = true
}


variable "public_key" {
  description = "SSH public key"
  type        = string  
  sensitive   = true
}

