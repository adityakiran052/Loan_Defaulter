provider "aws" {
  region = "us-east-1" # You can change this to your preferred region
}

resource "aws_instance" "loan_api_server" {
  ami           = "ami-0c55b159cbfafe1f0" # Standard Ubuntu AMI
  instance_type = "t2.micro"             # Free-tier eligible

  tags = {
    Name = "LoanDefaultPredictionServer"
  }
}
