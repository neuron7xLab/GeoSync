aws_region  = "us-east-1"
environment = "staging"
cluster_name = "tradepulse-staging"

node_groups = {
  general = {
    min_size       = 2
    max_size       = 6
    desired_size   = 3
    instance_types = ["m6i.large"]
    capacity_type  = "ON_DEMAND"
    labels = {
      "workload" = "api"
    }
    taints = []
  }
}

tags = {
  "CostCenter" = "tradepulse-staging"
}
