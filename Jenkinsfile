pipeline {
    agent any
    
    environment {
        DOCKER_COMPOSE_VERSION = '2.15.1'
        DOCKERHUB_CREDENTIALS = credentials('dockerhub')
        // Change this to your Docker Hub username or organization
        DOCKER_HUB_ACCOUNT = 'yourusername'
        IMAGE_TAG = "${env.BUILD_NUMBER}"
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Build') {
            steps {
                sh 'docker-compose build'
            }
        }
        
        stage('Test') {
            steps {
                sh '''
                # Start the services in detached mode
                docker-compose up -d
                
                # Wait for the backend to be ready (adjust the sleep time as needed)
                sleep 30
                
                # Run tests against the backend
                docker-compose exec -T backend python manage.py test
                
                # Shut down services
                docker-compose down
                '''
            }
        }
        
        stage('Prepare for Push') {
            steps {
                sh '''
                # Tag the images for Docker Hub
                docker tag stock-predictions-backend:latest ${DOCKER_HUB_ACCOUNT}/stock-predictions-backend:${IMAGE_TAG}
                docker tag stock-predictions-frontend:latest ${DOCKER_HUB_ACCOUNT}/stock-predictions-frontend:${IMAGE_TAG}
                
                # Also create latest tags
                docker tag stock-predictions-backend:latest ${DOCKER_HUB_ACCOUNT}/stock-predictions-backend:latest
                docker tag stock-predictions-frontend:latest ${DOCKER_HUB_ACCOUNT}/stock-predictions-frontend:latest
                '''
            }
        }
        
        stage('Push to Docker Hub') {
            steps {
                sh '''
                echo $DOCKERHUB_CREDENTIALS_PSW | docker login -u $DOCKERHUB_CREDENTIALS_USR --password-stdin
                
                # Push the tagged images to Docker Hub
                docker push ${DOCKER_HUB_ACCOUNT}/stock-predictions-backend:${IMAGE_TAG}
                docker push ${DOCKER_HUB_ACCOUNT}/stock-predictions-frontend:${IMAGE_TAG}
                docker push ${DOCKER_HUB_ACCOUNT}/stock-predictions-backend:latest
                docker push ${DOCKER_HUB_ACCOUNT}/stock-predictions-frontend:latest
                
                docker logout
                '''
            }
        }
        
        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            steps {
                sshagent(['production-server-ssh']) {
                    sh '''
                        echo "Deploying to production server..."
                        # Replace with your actual server details
                        ssh -o StrictHostKeyChecking=no user@your-production-server-ip "cd /path/to/your/project && \
                        export DOCKER_HUB_ACCOUNT=${DOCKER_HUB_ACCOUNT} && \
                        ./deploy.sh"
                    '''
                }
            }
        }
    }
    
    post {
        always {
            // Clean up
            sh 'docker-compose down || true'
            sh 'docker system prune -f || true'
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed!'
        }
    }
}