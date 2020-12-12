require('@tensorflow/tfjs-node');

const {Canvas, Image, ImageData, loadImage} = require("canvas");
const faceapi = require("face-api.js");
const jimp = require("jimp");
faceapi.env.monkeyPatch({Canvas, Image, ImageData});
const path = require("path");
const ssdMobilenetPath = path.join(__dirname, "./models/ssd_mobilenetv1");
const faceLandmarkPath = path.join(__dirname, "./models/face_landmark_68");
const faceRecognitionPath = path.join(__dirname, "./models/face_recognition");


/**
 * Get face api object (with models loaded) (async)
 * @returns {Object} Face api object 
 */
async function getFaceApiAsync(){
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(ssdMobilenetPath);
    await faceapi.nets.faceLandmark68Net.loadFromDisk(faceLandmarkPath);
    await faceapi.nets.faceRecognitionNet.loadFromDisk(faceRecognitionPath);
    return faceapi;
}

/**
 * Get face api object (with models loaded)
 * @param {CallableFunction<Error,Object>} callback 
 */
function getFaceApi(callback){
    faceapi.nets.ssdMobilenetv1.loadFromDisk(ssdMobilenetPath)
    .then(_ => {
        faceapi.nets.faceLandmark68Net.loadFromDisk(faceLandmarkPath)
        .then(_ => {
            faceapi.nets.faceRecognitionNet.loadFromDisk(faceRecognitionPath)
            .then(_ => {    
                callback(null, faceapi)
            }).catch(error => callback(error));
        }).catch(error => callback(error));
    }).catch(error => callback(error));
}

// get mime type and url base 64
function getMimeAndUrl(url){
    let base64ContentArray = url.split(",");    
    return [base64ContentArray[0].match(/[^:\s*]\w+\/[\w-+\d.]+(?=[;| ])/)[0],base64ContentArray[1]]
}

// gets the maximum amount that can be subtracted from a number without being negative
function getRelativePos(pos, max=20){
    for(var i=max; i>=0 ; i--) if(pos>=i) return i;
}

/**
 * Get faces from photo
 * @param {Object} api - Face api object
 * @param {URL} image - base 64 image
 * @param {CallableFunction<Error,Array<URL>>} callback Faces 
 */
function getFaces(api, image, callback){
    loadImage(image)
    .then(img_canvas => {
        api.detectAllFaces(img_canvas)
        .then(detections => {
            const [mimetype, url] = getMimeAndUrl(image);
            const buffer = Buffer.from(url, 'base64')  
            jimp.read(buffer)
            .then(img_jimp => {
                const processed_images = [];
                for(var detection of detections){
                    const index = detections.lastIndexOf(detection);
                    const box = detection.box;
                    const relative_x = getRelativePos(box._x);
                    const relative_y = getRelativePos(box._y);
                    new jimp((box._width + relative_x) * 2.25, (box._height + relative_y) * 2.25, "#000000", (err, frame) => {
                        if(err) callback(err);
                        else{
                            var copy_img = img_jimp.clone();
                            copy_img.crop(box._x - relative_x, box._y - relative_y, (box._width + relative_x) * 1.25, (box._height + relative_y) * 1.25);
                            frame.composite(copy_img, copy_img.getWidth() / 4, copy_img.getHeight() / 4).getBase64(mimetype, (err, value) => {
                                if(err) callback(err);
                                else{
                                    processed_images.push(value);
                                    if(index + 1 >= detections.length) callback(null, processed_images);
                                }
                            })
                        }
                    })
                }
            }).catch(err => callback(err));
        }).catch(err => callback(err));
    }).catch(err => callback(err));
}


/**
 * Get faces from photo (async)
 * @param {Object} api - Face api object
 * @param {URL} image - base 64 image
 * @returns {Array<URL>} Faces 
 */
async function getFacesAsync(api, image){
    const img_canvas  = await loadImage(image);
    const detections = await api.detectAllFaces(img_canvas);
    const [mimetype, url] = getMimeAndUrl(image);
    const img_jimp = await jimp.read(Buffer.from(url, 'base64'));
    const processed_images = [];
    for(var detection of detections){
        const box = detection.box;
        const relative_x = getRelativePos(box._x);
        const relative_y = getRelativePos(box._y);
        var frame = new jimp((box._width + relative_x) * 2.25, (box._height + relative_y) * 2.25, "#000000");
        var copy_img = img_jimp.clone();
        copy_img.crop(box._x - relative_x, box._y - relative_y, (box._width + relative_x) * 1.25, (box._height + relative_y) * 1.25);
        const processed = await frame.composite(copy_img, copy_img.getWidth() / 4, copy_img.getHeight() / 4).getBase64Async(mimetype);
        processed_images.push(processed);
    }
    return processed_images;
}


/**
 * Compare two faces
 * @param {Object} api - Face api object
 * @param {URL} one - base64 url
 * @param {URL} two - base64 url
 * @param {CallableFunction<Error,Boolean>} callback true if is the same person
 */
function compareFaces(api, one, two, callback){
    Promise.all([
        loadImage(one),
        loadImage(two)
    ]).then(([image_one, image_two]) => {
        Promise.all([
            api.detectSingleFace(image_one).withFaceLandmarks().withFaceDescriptor(),
            api.detectSingleFace(image_two).withFaceLandmarks().withFaceDescriptor()
        ]).then(([results_one, results_two]) => {
            if(results_one && results_two){
                const faceMatcher = new api.FaceMatcher(results_one);
                callback(null, (faceMatcher.findBestMatch(results_two.descriptor)._label !== "unknown"))
            }else callback("No results!");
        }).catch(error => callback(error));
    }).catch(error => callback(error));
}


/**
 * Compare two faces (async)
 * @param {Object} api - Face api object
 * @param {URL} one - base64 url
 * @param {URL} two - base64 url
 * @returns {Boolean} true if is the same person
 */
async function compareFacesAsync(api, one, two){
    const image_one = await loadImage(one);
    const image_two = await loadImage(two);
    const results_one = await api.detectSingleFace(image_one).withFaceLandmarks().withFaceDescriptor();
    const results_two = await api.detectSingleFace(image_two).withFaceLandmarks().withFaceDescriptor();
    if(results_one && results_two){
        const faceMatcher = new api.FaceMatcher(results_one);
        return faceMatcher.findBestMatch(results_two.descriptor)._label !== "unknown";
    }else throw "No results!";
}

module.exports = {
    compareFaces,
    compareFacesAsync,
    getFaces,
    getFacesAsync,
    getFaceApi,
    getFaceApiAsync
}