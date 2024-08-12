FROM jupyter/tensorflow-notebook

USER root
# Copy action files into the container
COPY . .

# download requirements 
RUN pip install -r requirements.txt


# Expose port 5055
EXPOSE 5000

# Define the command to run the actions server
CMD ["python", "app.py"]