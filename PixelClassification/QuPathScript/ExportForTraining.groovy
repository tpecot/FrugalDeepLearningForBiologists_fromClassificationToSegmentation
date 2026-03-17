import qupath.lib.images.servers.LabeledImageServer

def imageData = getCurrentImageData()

// Define output path (relative to project)
def pathOutput = buildFilePath(PROJECT_BASE_DIR, 'training')
mkdirs(pathOutput)

// Convert to downsample
double downsample = 1

// Create an ImageServer where the pixels are derived from annotations
def labelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.WHITE) // Specify background label (usually 0 or 255)
    .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
    .addLabel('Epithelium', 1)      // Choose output labels (the order matters!)
    .addLabel('Stroma', 2)
    .multichannelOutput(false)  // If true, each label is a different channel (required for multiclass probability)
    .build()

// Create an exporter that requests corresponding tiles from the original & labeled image servers
new TileExporter(imageData)
    .downsample(downsample)     // Define export resolution
    .tileSize(256)              // Define size of each tile, in pixels
    .labeledServer(labelServer) // Define the labeled image server to use (i.e. the one we just built)
    .annotatedTilesOnly(true)  // If true, only export tiles if there is a (labeled) annotation present
    .overlap(0)                // Define overlap, in pixel units at the export resolution
    .channels(0,1)
    .labeledImageExtension(".tif")
    .imageExtension(".tif")
    .imageSubDir('images')
    .labeledImageSubDir('masks')
    .writeTiles(pathOutput)     // Write tiles to the specified directory

def pixel_width = 0
def project = getProject()
for (entry in project.getImageList()) {
    def server = imageData.getServer()
    pixel_width = server.getPixelCalibration()
}


print 'Done!'