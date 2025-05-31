#!/usr/bin/env bash

minikube start --memory 4096 --cpus 4

kubectl delete secret regcred -n default 2>/dev/null || true
kubectl create secret docker-registry regcred \
  --docker-server="git.culab.ru:5050" \
  --docker-username="g.tebelev" \
  --docker-password="glpat-oSHmhjzeWgc8YcVP12Sf" -n default

kubectl apply -f manifests/model.yaml

kubectl rollout status deployment/inference -n default

minikube tunnel &

echo "Service available at: $(minikube service inference-service --url -n default)"

# kubectl port-forward svc/inference-service 1999:33999, тогда сервис будет доступен из браузера