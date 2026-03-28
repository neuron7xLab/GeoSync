output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "EKS API server endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate authority data"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

output "oidc_provider_arn" {
  description = "OIDC provider ARN for the cluster"
  value       = module.eks.oidc_provider_arn
}

output "node_group_roles" {
  description = "IAM roles associated with managed node groups"
  value       = { for name, group in module.eks.eks_managed_node_groups : name => group.iam_role_name }
}
