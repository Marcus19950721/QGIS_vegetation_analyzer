from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorDestination,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterEnum,
    QgsFeatureSink,
    QgsFeature,
    QgsFields,
    QgsField,
    QgsWkbTypes,
    QgsRectangle,
    QgsVectorLayer,
    QgsGeometry
)
import numpy as np
from osgeo import gdal
import processing

try:
    from skimage.morphology import remove_small_objects, binary_opening, disk
    from skimage.measure import label, regionprops
    from skimage.filters import gaussian
    ADVANCED_LIBS_INSTALLED = True
except ImportError:
    ADVANCED_LIBS_INSTALLED = False

class ModularVegetationAnalyzerAlgorithm(QgsProcessingAlgorithm):
    INPUT_RGB = 'INPUT_RGB'
    INPUT_DSM = 'INPUT_DSM'
    INPUT_DTM = 'INPUT_DTM'
    OUTPUT_VEG = 'OUTPUT_VEG'
    OUTPUT_CLASSIFIED = 'OUTPUT_CLASSIFIED'
    OUTPUT_VI_RASTER = 'OUTPUT_VI_RASTER'
    HEIGHT_THRESHOLD = 'HEIGHT_THRESHOLD'
    VI_METHOD = 'VI_METHOD'
    SENSITIVITY = 'SENSITIVITY'
    MIN_SIZE = 'MIN_SIZE'
    MANUAL_THRESHOLD = 'MANUAL_THRESHOLD'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return ModularVegetationAnalyzerAlgorithm()

    def name(self):
        return 'modular_vegetation_analyzer'

    def displayName(self):
        return self.tr('Modular Vegetation Analyzer')

    def group(self):
        return self.tr('Vegetation Analysis')

    def groupId(self):
        return 'vegetation_analysis'

    def shortHelpString(self):
        return self.tr("Detects vegetation from RGB and outputs vegetation index raster. If DSM/DTM are provided, it also classifies by height.")

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT_RGB,
            self.tr('1. Input RGB Orthomosaic (Required)')
        ))
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT_DSM,
            self.tr('2. Input DSM (Optional, for 3D Classification)'),
            optional=True
        ))
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT_DTM,
            self.tr('3. Input DTM (Optional, for 3D Classification)'),
            optional=True
        ))
        self.addParameter(QgsProcessingParameterEnum(
            self.VI_METHOD,
            self.tr('4. Detection: Vegetation Index'),
            ['ExG (Recommended)', 'VARI', 'GLI'],
            defaultValue=0
        ))
        self.addParameter(QgsProcessingParameterEnum(
            self.SENSITIVITY,
            self.tr('5. Detection: Sensitivity'),
            ['Low', 'Medium', 'High'],
            defaultValue=1
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_SIZE,
            self.tr('6. Detection: Minimum Object Size (sq meters)'),
            QgsProcessingParameterNumber.Double,
            defaultValue=0.1,
            minValue=0.01
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.MANUAL_THRESHOLD,
            self.tr('7. Detection: Manual Threshold (overrides sensitivity)'),
            QgsProcessingParameterNumber.Double,
            defaultValue=0.0,
            optional=True
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.HEIGHT_THRESHOLD,
            self.tr('8. Classification: Tree Height Threshold (meters)'),
            QgsProcessingParameterNumber.Double,
            defaultValue=4.0
        ))
        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT_VI_RASTER,
            self.tr('9. Output: Vegetation Index Raster'),
            optional=True
        ))
        self.addParameter(QgsProcessingParameterVectorDestination(
            self.OUTPUT_VEG,
            self.tr('10. Output: All Detected Vegetation'),
            optional=True
        ))
        self.addParameter(QgsProcessingParameterVectorDestination(
            self.OUTPUT_CLASSIFIED,
            self.tr('11. Output: Classified Vegetation (Trees/Bushes)'),
            optional=True
        ))

    def processAlgorithm(self, parameters, context, feedback):
        if not ADVANCED_LIBS_INSTALLED:
            feedback.reportError("CRITICAL ERROR: This tool requires 'scikit-image'. Please install it.")
            return {}

        rgb_layer = self.parameterAsRasterLayer(parameters, self.INPUT_RGB, context)
        dsm_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DSM, context)
        dtm_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DTM, context)
        vi_method = self.parameterAsEnum(parameters, self.VI_METHOD, context)
        sensitivity = self.parameterAsEnum(parameters, self.SENSITIVITY, context)
        min_size = self.parameterAsDouble(parameters, self.MIN_SIZE, context)
        manual_threshold = self.parameterAsDouble(parameters, self.MANUAL_THRESHOLD, context)
        height_threshold = self.parameterAsDouble(parameters, self.HEIGHT_THRESHOLD, context)

        if not self._validate_inputs(feedback, rgb_layer, dsm_layer, dtm_layer):
            return {}

        # Step 1: Calculate vegetation index
        feedback.pushInfo("Step 1/4: Calculating vegetation index...")
        vi_raster_ds, vi_array = self._calculate_vi(rgb_layer, vi_method, feedback)

        if parameters.get(self.OUTPUT_VI_RASTER):
            output_vi_path = self.parameterAsOutputLayer(parameters, self.OUTPUT_VI_RASTER, context)
            driver = gdal.GetDriverByName('GTiff')
            out_ds = driver.CreateCopy(output_vi_path, vi_raster_ds)
            out_ds.FlushCache()
            out_ds = None
            feedback.pushInfo(f"Saved vegetation index raster to: {output_vi_path}")

        # Step 2: Detect vegetation
        feedback.pushInfo("Step 2/4: Detecting vegetation...")
        veg_polygons = self._detect_vegetation(vi_array, rgb_layer, sensitivity, min_size, manual_threshold, feedback)
        if veg_polygons is None:
            return {}

        results = {}

        if parameters.get(self.OUTPUT_VEG):
            feedback.pushInfo("Step 3/4: Saving vegetation polygons...")
            sink_tuple = self.parameterAsSink(parameters, self.OUTPUT_VEG, context,
                                              veg_polygons.fields(), QgsWkbTypes.Polygon,
                                              rgb_layer.crs())
            if sink_tuple:
                sink, dest_id = sink_tuple
                feature_count = 0
                for feature in veg_polygons.getFeatures():
                    sink.addFeature(feature, QgsFeatureSink.FastInsert)
                    feature_count += 1
                feedback.pushInfo(f"Saved {feature_count} vegetation polygons")
                results[self.OUTPUT_VEG] = dest_id

        # Step 3: Height-based classification (if DSM/DTM provided)
        if all([dsm_layer, dtm_layer, parameters.get(self.OUTPUT_CLASSIFIED)]):
            feedback.pushInfo("Step 4/4: Classifying vegetation by height...")
            try:
                chm_result = processing.run('qgis:rastercalculator', {
                    'EXPRESSION': '\"A@1\" - \"B@1\"',
                    'LAYERS': [dsm_layer, dtm_layer],
                    'OUTPUT': 'TEMPORARY_OUTPUT'
                }, context=context, feedback=feedback, is_child_algorithm=True)

                if not chm_result or 'OUTPUT' not in chm_result:
                    feedback.reportError("Failed to calculate Canopy Height Model")
                    return results

                chm_layer = context.takeResultLayer(chm_result['OUTPUT'])

                zonal_result = processing.run('qgis:zonalstatistics', {
                    'INPUT': veg_polygons,
                    'RASTER': chm_layer,
                    'STATS': [2],  # max
                    'COLUMN_PREFIX': 'height_',
                    'OUTPUT': 'memory:'
                }, context=context, feedback=feedback, is_child_algorithm=True)

                veg_with_height = zonal_result['OUTPUT']

                fields = QgsFields()
                fields.append(QgsField('area_sqm', QVariant.Double))
                fields.append(QgsField('height_m', QVariant.Double))
                fields.append(QgsField('class', QVariant.String))

                sink_tuple = self.parameterAsSink(parameters, self.OUTPUT_CLASSIFIED, context,
                                                  fields, QgsWkbTypes.Point, rgb_layer.crs())

                if sink_tuple:
                    sink, dest_id = sink_tuple
                    tree_count = bush_count = 0

                    for feature in veg_with_height.getFeatures():
                        height = feature['height_max'] or 0
                        area = feature.geometry().area()
                        class_name = 'Tree' if height >= height_threshold else 'Bush'

                        new_feature = QgsFeature()
                        new_feature.setGeometry(feature.geometry().centroid())
                        new_feature.setAttributes([
                            round(area, 2),
                            round(height, 2),
                            class_name
                        ])
                        sink.addFeature(new_feature, QgsFeatureSink.FastInsert)

                        if class_name == 'Tree':
                            tree_count += 1
                        else:
                            bush_count += 1

                    feedback.pushInfo(f"Classified {tree_count} trees and {bush_count} bushes")
                    results[self.OUTPUT_CLASSIFIED] = dest_id

            except Exception as e:
                feedback.reportError(f"Error in classification: {str(e)}")
                return results
        else:
            feedback.pushInfo("Skipping classification (no DSM/DTM or output specified)")

        return results

    def _validate_inputs(self, feedback, rgb, dsm, dtm):
        if not rgb:
            feedback.reportError("Input RGB Orthomosaic is missing.")
            return False
        if (dsm and not dtm) or (dtm and not dsm):
            feedback.reportError("For 3D classification, BOTH DSM and DTM must be provided.")
            return False
        if dsm and dtm:
            if not (rgb.crs() == dsm.crs() == dtm.crs()):
                feedback.reportError("CRS mismatch between inputs.")
                return False
            if not (dsm.width() == dtm.width() and dsm.height() == dtm.height()):
                feedback.reportError("Raster dimension mismatch between DSM and DTM.")
                return False
        if rgb.bandCount() < 3:
            feedback.reportError(f"RGB input must have 3 bands, but has {rgb.bandCount()}.")
            return False
        return True

    def _calculate_vi(self, rgb_layer, vi_method, feedback):
        raster_path = rgb_layer.source()
        ds = gdal.Open(raster_path)
        r, g, b = (ds.GetRasterBand(i).ReadAsArray().astype(np.float32) for i in range(1, 4))

        if vi_method == 0:  # ExG
            vi = 2 * g - r - b
            method_name = "ExG"
        elif vi_method == 1:  # VARI
            vi = np.divide(g - r, g + r - b, out=np.zeros_like(g), where=(g + r - b) != 0)
            method_name = "VARI"
        else:  # GLI
            vi = np.divide(2 * g - r - b, 2 * g + r + b, out=np.zeros_like(g), where=(2 * g + r + b) != 0)
            method_name = "GLI"

        feedback.pushInfo(f"Calculated {method_name} vegetation index")

        driver = gdal.GetDriverByName('MEM')
        vi_ds = driver.Create('', ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Float32)
        vi_ds.SetGeoTransform(ds.GetGeoTransform())
        vi_ds.SetProjection(ds.GetProjection())
        vi_ds.GetRasterBand(1).WriteArray(vi)

        ds = None
        return vi_ds, vi

    def _detect_vegetation(self, vi, rgb_layer, sensitivity, min_size, manual_threshold, feedback):
        if manual_threshold > 0:
            threshold = manual_threshold
        else:
            sensitivity_factors = [1.3, 1.0, 0.7]  # Low, Medium, High
            threshold_factor = sensitivity_factors[sensitivity]

            positive_vi = vi[vi > 0]
            if len(positive_vi) == 0:
                feedback.reportError("No positive vegetation index values found")
                return None
            base_threshold = np.percentile(positive_vi, 70)
            threshold = base_threshold * threshold_factor

        mask = vi > threshold
        pixel_size = abs(rgb_layer.rasterUnitsPerPixelX() * rgb_layer.rasterUnitsPerPixelY())
        min_pixels = max(1, int(min_size / pixel_size))

        mask = gaussian(mask, sigma=0.8) > 0.5
        mask = binary_opening(mask, disk(2))
        mask = remove_small_objects(mask, min_size=min_pixels)

        uri = f'Polygon?crs={rgb_layer.crs().authid()}&field=area_sqm:double&field=vi_mean:double'
        vector_layer = QgsVectorLayer(uri, 'vegetation', 'memory')
        provider = vector_layer.dataProvider()

        regions = regionprops(label(mask))
        for region in regions:
            area = float(region.area) * pixel_size
            if area < min_size:
                continue

            minr, minc, maxr, maxc = region.bbox
            x1 = rgb_layer.extent().xMinimum() + minc * rgb_layer.rasterUnitsPerPixelX()
            y1 = rgb_layer.extent().yMaximum() - maxr * rgb_layer.rasterUnitsPerPixelY()
            x2 = rgb_layer.extent().xMinimum() + maxc * rgb_layer.rasterUnitsPerPixelX()
            y2 = rgb_layer.extent().yMaximum() - minr * rgb_layer.rasterUnitsPerPixelY()

            rect = QgsRectangle(x1, y1, x2, y2)
            geom = QgsGeometry.fromRect(rect)

            region_mask = region.image
            region_vi = vi[minr:maxr, minc:maxc][region_mask]
            vi_mean = float(np.mean(region_vi))

            feat = QgsFeature()
            feat.setGeometry(geom)
            feat.setAttributes([area, vi_mean])
            provider.addFeature(feat)

        vector_layer.updateExtents()
        return vector_layer
