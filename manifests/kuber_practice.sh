kubectl get pods -n my-namespace -l app=app1
kubectl get pods --all-namespaces -l app=app1
kubectl get deployment app3 -n my-namespace
kubectl logs -n my-namespace deployments/app2
kubectl scale deployment app1 --replicas=4 -n my-namespace
kubectl scale deployment app1 --replicas=1 -n my-namespace
kubectl delete pods -n my-namespace -l app=app1
kubectl expose deployment app1 --type=LoadBalancer --name=app1-service --port=33888 --target-port=80 -n my-namespace