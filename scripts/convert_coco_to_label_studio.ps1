param(
    [Parameter(Mandatory = $true)][string]$CocoPath,
    [Parameter(Mandatory = $true)][string]$ImagesRoot,
    [Parameter(Mandatory = $true)][string]$OutputPath
)

$coco = Get-Content -LiteralPath $CocoPath -Raw | ConvertFrom-Json
$categories = @{}
foreach ($category in $coco.categories) {
    $categories[[int]$category.id] = $category.name
}

$annotationsByImage = @{}
foreach ($annotation in $coco.annotations) {
    $key = [int]$annotation.image_id
    if (-not $annotationsByImage.ContainsKey($key)) {
        $annotationsByImage[$key] = [System.Collections.Generic.List[object]]::new()
    }
    $annotationsByImage[$key].Add($annotation)
}

$tasks = foreach ($image in $coco.images) {
    $imageId = [int]$image.id
    $imageAnnotations = if ($annotationsByImage.ContainsKey($imageId)) {
        $annotationsByImage[$imageId]
    } else {
        @()
    }
    $results = foreach ($annotation in $imageAnnotations) {
        $bbox = $annotation.bbox
        [ordered]@{
            from_name = 'label'
            to_name = 'image'
            type = 'rectanglelabels'
            original_width = [int]$image.width
            original_height = [int]$image.height
            image_rotation = 0
            value = [ordered]@{
                x = [math]::Round((100.0 * [double]$bbox[0] / [double]$image.width), 6)
                y = [math]::Round((100.0 * [double]$bbox[1] / [double]$image.height), 6)
                width = [math]::Round((100.0 * [double]$bbox[2] / [double]$image.width), 6)
                height = [math]::Round((100.0 * [double]$bbox[3] / [double]$image.height), 6)
                rotation = 0
                rectanglelabels = @($categories[[int]$annotation.category_id])
            }
        }
    }

    $imagePath = Join-Path $ImagesRoot $image.file_name
    # Prefer Label Studio local-files URLs (browsers block file://).
    # DOCUMENT_ROOT should be the parent of the relative path used below.
    $docRoot = $env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT
    if ($docRoot) {
        $full = [System.IO.Path]::GetFullPath($imagePath)
        $rootFull = [System.IO.Path]::GetFullPath($docRoot).TrimEnd('\', '/')
        if ($full.StartsWith($rootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
            $rel = $full.Substring($rootFull.Length).TrimStart('\', '/').Replace('\', '/')
            $imageUrl = "/data/local-files/?d=$rel"
        } else {
            $imageUrl = ('file:///' + (($imagePath -replace '\\', '/') -replace ' ', '%20'))
        }
    } else {
        $imageUrl = ('file:///' + (($imagePath -replace '\\', '/') -replace ' ', '%20'))
    }
    [ordered]@{
        data = [ordered]@{ image = $imageUrl }
        predictions = @([ordered]@{
            model_version = 'roboflow-sam3-auto-label'
            score = 1.0
            result = @($results)
        })
    }
}

$json = $tasks | ConvertTo-Json -Depth 12
[System.IO.File]::WriteAllText($OutputPath, $json, [System.Text.UTF8Encoding]::new($false))
Write-Output ("Created {0} Label Studio tasks." -f @($tasks).Count)
