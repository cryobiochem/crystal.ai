<?php // TODO: quando fazes upload, o ficheiro não vai para o diretório inputs, e também não deteta erro se meteres um pdf ?>

<?php
$target_dir = "inputs/";
$target_file = $target_dir . basename($_FILES["imageInput"]["name"]);
$uploadOk = 1;
$imageFileType = strtolower(pathinfo($target_file,PATHINFO_EXTENSION));

  // Check if image file is a actual image or fake image
  if(isset($_POST["submit"])) {
      $check = getimagesize($_FILES["imageInput"]["tmp_name"]);
      if($check !== false) {
          echo "File is an image - " . $check["mime"] . ".";
          $uploadOk = 1;
      } else {
          echo "File is not an image.";
          $uploadOk = 0;
  }

  // Check file size
  if ($_FILES["imageInput"]["size"] > 25000000) {
      echo "Sorry, your image is larger than 25 Mb. Rescale and submit again";
      $uploadOk = 0;
  }

  // Allow only certain file formats
  if($imageFileType != "jpg" && $imageFileType != "png" && $imageFileType != "jpeg"
  && $imageFileType != "gif" && $imageFileType != "svg" ) {
      echo "Sorry, only JPG, JPEG, PNG, GIF & SVG files work.";
      $uploadOk = 0;
  }

  // If file already exists, change name
  if (file_exists($target_file)) {
      echo "Sorry, file already exists in our database. Please change image name.";
      $uploadOk = 0;
  }

  // Check if $uploadOk is set to 0 by an error
  if ($uploadOk == 0) {
      echo "Sorry, there was an error uploading your image.";

  // if everything is ok, try to upload file
  } else {
      if (move_uploaded_file($_FILES["imageInput"]["tmp_name"], $target_file)) {
          echo "The image ". basename( $_FILES["imageInput"]["name"]). " has been uploaded.";
      } else {
          echo "Sorry, there was an error uploading your image.";
      }
  }

}
?>
