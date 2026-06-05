output "s3_bucket_name" {
  value = aws_s3_bucket.artifacts.bucket
}

output "iam_access_key_id" {
  value = aws_iam_access_key.mlops.id
}

output "iam_secret_access_key" {
  value     = aws_iam_access_key.mlops.secret
  sensitive = true
}
