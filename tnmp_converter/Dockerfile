FROM debian:bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ============================================================
# System dependencies
# ============================================================

RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip \
        python3-dev \
        apache2 \
        libapache2-mod-wsgi-py3 \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# Application
# ============================================================

WORKDIR /var/www/tnmp_converter


# ============================================================
# Python dependencies
# ============================================================

COPY requirements.txt .

RUN pip3 install \
    --break-system-packages \
    --no-cache-dir \
    -r requirements.txt


# ============================================================
# Copy application
# ============================================================

COPY . /var/www/tnmp_converter


# ============================================================
# Create directories
# ============================================================

RUN mkdir -p \
    /var/www/tnmp_converter/static/uploads \
    /var/www/tnmp_converter/static/generated


# ============================================================
# Apache configuration
# ============================================================

COPY apache/tnmp_converter.conf \
    /etc/apache2/sites-available/tnmp_converter.conf

COPY tnmp_converter.wsgi \
    /var/www/tnmp_converter/tnmp_converter.wsgi


# ============================================================
# Enable Apache modules/site
# ============================================================

RUN a2enmod wsgi && \
    a2dissite 000-default.conf && \
    a2ensite tnmp_converter.conf

# ============================================================
# Permissions
# ============================================================

RUN chown -R www-data:www-data /var/www/tnmp_converter && \
    chmod -R 755 /var/www/tnmp_converter


# ============================================================
# Port
# ============================================================

EXPOSE 80


# ============================================================
# Start Apache
# ============================================================

CMD ["apachectl", "-D", "FOREGROUND"]

