# نسخه پایتون رو اینجا مشخص کن (مثلاً python:3.10-slim یا python:3.9-slim — توصیه من 3.10 برای Rasa 3.6.21)
FROM python:3.10-slim

# وابستگی‌های سیستم رو نصب کن (برای Rasa و TensorFlow لازم هست)
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# دایرکتوری کاری رو تنظیم کن
WORKDIR /app

# pip رو آپدیت کن و Rasa رو نصب کن (آخرین نسخه پایدار: 3.6.21)
RUN pip install --upgrade pip
RUN pip install rasa==3.6.21

# فایل‌های پروژه رو کپی کن (بعداً اضافه می‌شه)
COPY . /app

# ENTRYPOINT رو اضافه کن تا کامندها مثل "init" مستقیماً با "rasa" اجرا بشن (فیکس خطای شما)
ENTRYPOINT ["rasa"]

# پورت‌های Rasa رو اکسپوز کن
EXPOSE 5005 5055

# کامند پیش‌فرض: Rasa سرور رو اجرا کن
CMD ["run", "--enable-api", "--cors", "*", "--debug"]