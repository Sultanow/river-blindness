# input
REGION=us-east-1

# evaulated
ECR_HOST=954362353459.dkr.ecr.$(REGION).amazonaws.com
REPO=sm-training-custom
TAG=torch-1.8.1-cu111-noGPL
SM_IMAGE=gdsc5-smstudio-custom

get-vars:
	$(eval DOMAINID := $(shell aws --region $(REGION) sagemaker list-domains --query 'Domains[0].DomainId'))
	$(eval SMROLE := $(shell aws --region $(REGION) sagemaker describe-domain --domain-id ${DOMAINID} --query 'DefaultUserSettings.ExecutionRole'))

test-vars: get-vars
	echo $(DOMAINID)
	echo $(SMROLE)

update-domain: get-vars
	aws --region $(REGION) sagemaker update-domain --domain-id $(DOMAINID) --cli-input-json file://default-user-settings.json 1>/dev/null

create-image: get-vars
	aws --region $(REGION) sagemaker create-image --image-name $(SM_IMAGE) --role-arn $(SMROLE) 1>/dev/null
	aws --region $(REGION) sagemaker create-image-version --image-name $(SM_IMAGE) \
		--base-image $(ECR_HOST)/$(REPO):$(TAG) 1>/dev/null
	aws --region $(REGION) sagemaker create-app-image-config --cli-input-json file://app-image-config-input.json 1>/dev/null
	make update-domain
	
update-image: get-vars
	aws --region $(REGION) sagemaker create-image-version --image-name $(SM_IMAGE) \
		--base-image $(ECR_HOST)/$(REPO):$(TAG) 1>/dev/null
	aws --region $(REGION) sagemaker update-app-image-config --cli-input-json file://app-image-config-input.json 1>/dev/null
	make update-domain
