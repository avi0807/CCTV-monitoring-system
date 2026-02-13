import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
import json
from enum import Enum
import base64
import os
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


# ============================================================
# CONFIG
# ============================================================
import os
from dotenv import load_dotenv

# Load API key from environment variable or .env file
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in environment variables")
    print("Please set it in .env file or environment variable")
    print("Example: export GEMINI_API_KEY='your-key-here'")
    GEMINI_API_KEY = input("Or enter your API key now: ")


# ============================================================
# ENUMS / DATA MODELS
# ============================================================

class SpaceType(Enum):
    AISLE = "aisle"
    ENTRANCE = "entrance"
    CHECKOUT = "checkout"
    BACKROOM = "backroom"


class TrafficLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AlertPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass
class YOLODetection:
    """Raw YOLO detection output"""
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]


@dataclass
class CleanlinessAnalysis:
    """LLM's cleanliness assessment"""
    overall_cleanliness_score: float  # 0-10, where 10 is pristine
    floor_condition: str  # "clean", "lightly_dirty", "moderately_dirty", "very_dirty"
    visible_debris: List[str]  # List of debris types detected
    debris_locations: List[str]  # "floor", "shelves", "corners", etc.
    spills_present: bool
    stains_present: bool
    reasoning: str


@dataclass
class SpatialAnalysis:
    """LLM's spatial understanding"""
    floor_objects: List[Dict]  # Objects detected on floor with descriptions
    shelf_objects: List[Dict]  # Objects on shelves
    misplaced_items: List[Dict]  # Items that shouldn't be where they are
    spatial_reasoning: str


@dataclass
class MerchandiseAnalysis:
    """LLM's merchandise assessment"""
    shelf_fullness_score: float  # 0-10
    shelf_organization_score: float  # 0-10
    empty_spaces_count: int
    misplaced_products_count: int
    fallen_products_count: int
    reasoning: str


@dataclass
class AlertDecision:
    alert_required: bool
    priority: AlertPriority
    reasoning: str
    recommended_action: str
    estimated_time_minutes: int
    confidence_level: float  # 0-1


@dataclass
class AnalysisResult:
    cleanliness_analysis: Optional[CleanlinessAnalysis]
    spatial_analysis: Optional[SpatialAnalysis]
    merchandise_analysis: Optional[MerchandiseAnalysis]
    alert_decision: AlertDecision
    raw_detections: List[YOLODetection]


# ============================================================
# YOLO DETECTOR (SIMPLIFIED)
# ============================================================

class SimpleYOLODetector:
    """Only does object detection, no classification"""
    
    def __init__(self):
        print("Loading YOLO model...")
        self.model = YOLO("yolov8n.pt")
        print("YOLO model loaded")

    def detect_objects(self, image: np.ndarray) -> Tuple[List[YOLODetection], int, int]:
        """
        Detect objects in image using YOLO.
        Returns detections and image dimensions.
        """
        height, width = image.shape[:2]
        
        results = self.model(image, conf=0.25, verbose=False)
        
        detections = []
        for result in results:
            for box in result.boxes:
                bbox = box.xyxy[0].tolist()
                detections.append(
                    YOLODetection(
                        class_name=result.names[int(box.cls[0])],
                        confidence=float(box.conf[0]),
                        bbox=bbox
                    )
                )
        
        return detections, width, height


# ============================================================
# VISION LLM ANALYZER (DOES ALL THE SMART WORK)
# ============================================================

class VisionLLMAnalyzer:
    """Uses Gemini Vision to analyze images intelligently"""
    
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=api_key,
            temperature=0.1
        )
        print("Vision LLM initialized (gemini-2.5-flash)")

    def analyze_cleanliness(
        self,
        image: np.ndarray,
        detections: List[YOLODetection],
        space_type: SpaceType,
        traffic_level: TrafficLevel,
        hours_since_cleaned: float
    ) -> Tuple[CleanlinessAnalysis, SpatialAnalysis]:
        """
        Send image + detection info to LLM for cleanliness analysis.
        LLM does the actual visual inspection and spatial reasoning.
        """
        
        # Encode image
        image_base64 = self._encode_image(image)
        
        # Build detection summary
        detection_summary = self._format_detections(detections)
        
        prompt = f"""You are an expert facility maintenance inspector analyzing a retail store image for cleanliness.

**DETECTED OBJECTS (from YOLO):**
{detection_summary}

**CONTEXT:**
- Space type: {space_type.value}
- Traffic level: {traffic_level.value}
- Hours since last cleaning: {hours_since_cleaned:.1f}

**YOUR TASK:**
Carefully examine this image and provide a detailed cleanliness assessment.

Look for:
1. **Floor condition**: Is the floor clean or dirty? Look for:
   - Visible debris (paper, wrappers, dirt, dust)
   - Spills or wet spots
   - Stains or discoloration
   - Scuff marks or grime buildup
   - Overall floor appearance

2. **Spatial analysis**: For each detected object, determine:
   - Is it on the floor or on a shelf/display?
   - Is it in the correct location?
   - Should it be there?

3. **Severity assessment**: Rate how dirty the space is on a scale of 0-10:
   - 0-2: Pristine, spotless
   - 3-4: Very clean, minor dust
   - 5-6: Acceptable, some visible dirt
   - 7-8: Noticeably dirty, needs cleaning soon
   - 9-10: Very dirty, needs immediate attention

**IMPORTANT GUIDELINES:**
- Base your assessment on what you ACTUALLY SEE in the image, not just the YOLO detections
- YOLO might miss things or misclassify - use your visual understanding
- Consider lighting - don't confuse shadows with dirt
- Consider floor texture - some floors naturally look darker
- Be accurate but practical - small dust is normal in high-traffic areas

**RESPONSE FORMAT:**
Respond with ONLY a valid JSON object. Do not include any text before or after the JSON.
Use boolean true/false (not strings), numbers for scores, and arrays for lists.

CRITICAL: Each item in floor_objects, shelf_objects, and misplaced_items MUST be an object (dict) with these fields:
- object: string (name of the object)
- location: string (where it is)
- should_be_here: boolean (for floor_objects) OR properly_placed: boolean (for shelf_objects)
- concern_level: string (for floor_objects: "low", "medium", "high")

Example response structure:
{{
    "cleanliness_analysis": {{
        "overall_cleanliness_score": 8.5,
        "floor_condition": "clean",
        "visible_debris": ["dust", "paper"],
        "debris_locations": ["corner", "under shelf"],
        "spills_present": false,
        "stains_present": false,
        "reasoning": "Floor appears well-maintained..."
    }},
    "spatial_analysis": {{
        "floor_objects": [
            {{"object": "bottle", "location": "center aisle", "should_be_here": false, "concern_level": "medium"}}
        ],
        "shelf_objects": [
            {{"object": "product", "location": "shelf 2", "properly_placed": true}}
        ],
        "misplaced_items": [
            {{"object": "box", "current_location": "floor", "should_be": "shelf"}}
        ],
        "spatial_reasoning": "All items properly placed..."
    }}
}}

Now provide your analysis:"""

        try:
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{image_base64}"
                    }
                ]
            )
            
            response = self.llm.invoke([message])
            print(f"\n[DEBUG] LLM Response for cleanliness (first 300 chars):\n{response.content[:300]}...\n")
            
            result = self._parse_json_response(response.content)
            
            cleanliness = CleanlinessAnalysis(
                overall_cleanliness_score=float(result["cleanliness_analysis"]["overall_cleanliness_score"]),
                floor_condition=result["cleanliness_analysis"]["floor_condition"],
                visible_debris=result["cleanliness_analysis"]["visible_debris"],
                debris_locations=result["cleanliness_analysis"]["debris_locations"],
                spills_present=bool(result["cleanliness_analysis"]["spills_present"]),
                stains_present=bool(result["cleanliness_analysis"]["stains_present"]),
                reasoning=result["cleanliness_analysis"]["reasoning"]
            )
            
            spatial = SpatialAnalysis(
                floor_objects=self._sanitize_items_list(result["spatial_analysis"]["floor_objects"]),
                shelf_objects=self._sanitize_items_list(result["spatial_analysis"]["shelf_objects"]),
                misplaced_items=self._sanitize_items_list(result["spatial_analysis"]["misplaced_items"]),
                spatial_reasoning=result["spatial_analysis"]["spatial_reasoning"]
            )
            
            return cleanliness, spatial
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error in cleanliness analysis: {e}")
            return self._fallback_cleanliness_analysis(), self._fallback_spatial_analysis()
        except KeyError as e:
            print(f"❌ Missing key in cleanliness response: {e}")
            return self._fallback_cleanliness_analysis(), self._fallback_spatial_analysis()
        except Exception as e:
            print(f"❌ Error in LLM analysis: {e}")
            print(f"Error type: {type(e).__name__}")
            return self._fallback_cleanliness_analysis(), self._fallback_spatial_analysis()

    def analyze_merchandise(
        self,
        image: np.ndarray,
        detections: List[YOLODetection],
        expected_shelf_fullness: float = 80.0
    ) -> MerchandiseAnalysis:
        """Analyze merchandise presentation and organization"""
        
        image_base64 = self._encode_image(image)
        detection_summary = self._format_detections(detections)
        
        prompt = f"""You are a retail merchandising expert analyzing shelf presentation.

**DETECTED OBJECTS:**
{detection_summary}

**EXPECTED SHELF FULLNESS:** {expected_shelf_fullness}%

**YOUR TASK:**
Examine the image and assess:

1. **Shelf fullness**: How well-stocked are the shelves? (0-10 scale)
2. **Organization**: Are products neatly arranged? (0-10 scale)
3. **Empty spaces**: Count visible gaps in shelving
4. **Misplaced products**: Products in wrong locations
5. **Fallen products**: Items that have fallen on the floor

Respond in JSON:
{{
    "shelf_fullness_score": 0-10,
    "shelf_organization_score": 0-10,
    "empty_spaces_count": number,
    "misplaced_products_count": number,
    "fallen_products_count": number,
    "reasoning": "detailed explanation"
}}"""

        try:
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
                ]
            )
            
            response = self.llm.invoke([message])
            result = self._parse_json_response(response.content)
            
            return MerchandiseAnalysis(
                shelf_fullness_score=result["shelf_fullness_score"],
                shelf_organization_score=result["shelf_organization_score"],
                empty_spaces_count=result["empty_spaces_count"],
                misplaced_products_count=result["misplaced_products_count"],
                fallen_products_count=result["fallen_products_count"],
                reasoning=result["reasoning"]
            )
            
        except Exception as e:
            print(f"Error in merchandise analysis: {e}")
            return self._fallback_merchandise_analysis()

    def make_alert_decision(
        self,
        cleanliness: Optional[CleanlinessAnalysis],
        spatial: Optional[SpatialAnalysis],
        merchandise: Optional[MerchandiseAnalysis],
        space_type: SpaceType,
        traffic_level: TrafficLevel,
        store_tier: str,
        hours_since_cleaned: float
    ) -> AlertDecision:
        """Final decision on whether to raise an alert"""
        
        context = {
            "space_type": space_type.value,
            "traffic_level": traffic_level.value,
            "store_tier": store_tier,
            "hours_since_cleaned": hours_since_cleaned
        }
        
        if cleanliness:
            analysis_data = {
                "cleanliness_score": cleanliness.overall_cleanliness_score,
                "floor_condition": cleanliness.floor_condition,
                "visible_debris": cleanliness.visible_debris,
                "spills_present": cleanliness.spills_present,
                "stains_present": cleanliness.stains_present,
                "cleanliness_reasoning": cleanliness.reasoning
            }
            
            if spatial:
                analysis_data.update({
                    "floor_objects_count": len(spatial.floor_objects),
                    "misplaced_items_count": len(spatial.misplaced_items),
                    "spatial_reasoning": spatial.spatial_reasoning
                })
            
            decision_type = "cleanliness"
            
        elif merchandise:
            analysis_data = {
                "shelf_fullness": merchandise.shelf_fullness_score,
                "shelf_organization": merchandise.shelf_organization_score,
                "empty_spaces": merchandise.empty_spaces_count,
                "fallen_products": merchandise.fallen_products_count,
                "merchandise_reasoning": merchandise.reasoning
            }
            decision_type = "merchandise"
        else:
            return self._fallback_decision()
        
        prompt = f"""You are a retail operations manager making alert decisions.

**ANALYSIS TYPE:** {decision_type}

**ANALYSIS DATA:**
{json.dumps(analysis_data, indent=2)}

**CONTEXT:**
{json.dumps(context, indent=2)}

**DECISION GUIDELINES FOR CLEANLINESS:**
- Score 0-4: Clean, no alert needed
- Score 5-6: Monitor, alert if high-traffic area and not cleaned recently
- Score 7-8: Alert recommended, cleaning needed soon
- Score 9-10: Alert required, immediate cleaning needed

Consider:
- High traffic areas get dirtier faster
- Premium stores need higher standards
- Spills and safety hazards are immediate priorities
- Time since last cleaning matters

**DECISION GUIDELINES FOR MERCHANDISE:**
- Fallen products: Immediate alert (safety + shrink risk)
- Shelf fullness < 60%: Alert for restocking
- Poor organization: Alert if score < 5

CRITICAL: You must respond with ONLY a valid JSON object. No markdown, no explanations, no code blocks - just the JSON.

{{
    "alert_required": true,
    "priority": "medium",
    "reasoning": "your detailed reasoning here",
    "recommended_action": "specific action to take",
    "estimated_time_minutes": 15,
    "confidence_level": 0.85
}}

Respond now with your decision JSON:"""

        try:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Debug output
            print(f"\n[DEBUG] Raw LLM Response:")
            print(f"{response.content}")
            print(f"[DEBUG] Response type: {type(response.content)}")
            print()
            
            result = self._parse_json_response(response.content)
            
            return AlertDecision(
                alert_required=result["alert_required"],
                priority=AlertPriority(result["priority"]),
                reasoning=result["reasoning"],
                recommended_action=result["recommended_action"],
                estimated_time_minutes=int(result["estimated_time_minutes"]),
                confidence_level=float(result["confidence_level"])
            )
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error in decision making: {e}")
            print(f"Response was: {response.content}")
            return self._fallback_decision()
        except KeyError as e:
            print(f"❌ Missing key in response: {e}")
            print(f"Response was: {response.content}")
            score = cleanliness.overall_cleanliness_score if cleanliness else None
            return self._fallback_decision(score)
        except Exception as e:
            print(f"❌ Error in decision making: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(f"Full traceback:")
            traceback.print_exc()
            score = cleanliness.overall_cleanliness_score if cleanliness else None
            return self._fallback_decision(score)

    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image to base64"""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    
    def _sanitize_items_list(self, items: List) -> List[Dict]:
        """Ensure all items in list are dicts, convert strings to dicts"""
        sanitized = []
        for item in items:
            if isinstance(item, dict):
                sanitized.append(item)
            elif isinstance(item, str):
                # Convert string to dict format
                sanitized.append({
                    "object": item,
                    "current_location": "unknown",
                    "should_be": "unknown"
                })
            else:
                # Skip invalid items
                continue
        return sanitized

    def _format_detections(self, detections: List[YOLODetection]) -> str:
        """Format YOLO detections for LLM"""
        if not detections:
            return "No objects detected"
        
        summary = {}
        for d in detections:
            summary[d.class_name] = summary.get(d.class_name, 0) + 1
        
        lines = [f"- {count}x {obj} (confidence: varies)" for obj, count in summary.items()]
        return "\n".join(lines)

    def _parse_json_response(self, text: str) -> dict:
        """Extract and parse JSON from LLM response with robust error handling"""
        print(f"[DEBUG] Starting to parse response...")
        print(f"[DEBUG] Response length: {len(text)} characters")
        
        original_text = text
        text = text.strip()
        
        # Remove markdown code blocks if present
        if "```json" in text.lower():
            print("[DEBUG] Found ```json marker, extracting...")
            start_marker = text.lower().find("```json")
            if start_marker >= 0:
                text = text[start_marker + 7:]  # Skip ```json
                end_marker = text.find("```")
                if end_marker >= 0:
                    text = text[:end_marker]
        elif text.startswith("```"):
            print("[DEBUG] Found ``` marker, cleaning...")
            lines = text.split("\n")
            text = "\n".join([l for l in lines if not l.strip().startswith("```")])
        
        text = text.strip()
        print(f"[DEBUG] After cleaning: {text[:200]}...")
        
        # Find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        
        if start >= 0 and end > start:
            json_str = text[start:end]
            print(f"[DEBUG] Extracted JSON string: {json_str[:300]}...")
            try:
                result = json.loads(json_str)
                print(f"[DEBUG] ✅ Successfully parsed JSON!")
                print(f"[DEBUG] Keys found: {list(result.keys())}")
                return result
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
                print(f"❌ Error at position {e.pos}")
                print(f"❌ Attempted to parse: {json_str[:500]}...")
                raise
        else:
            print(f"❌ Could not find JSON object in response")
            print(f"❌ Start: {start}, End: {end}")
            print(f"❌ Full text: {text[:500]}...")
            raise ValueError("No JSON object found in response")

    def _fallback_cleanliness_analysis(self) -> CleanlinessAnalysis:
        return CleanlinessAnalysis(
            overall_cleanliness_score=5.0,
            floor_condition="unknown",
            visible_debris=[],
            debris_locations=[],
            spills_present=False,
            stains_present=False,
            reasoning="Error in analysis - manual review needed"
        )

    def _fallback_spatial_analysis(self) -> SpatialAnalysis:
        return SpatialAnalysis(
            floor_objects=[],
            shelf_objects=[],
            misplaced_items=[],
            spatial_reasoning="Error in analysis"
        )

    def _fallback_merchandise_analysis(self) -> MerchandiseAnalysis:
        return MerchandiseAnalysis(
            shelf_fullness_score=5.0,
            shelf_organization_score=5.0,
            empty_spaces_count=0,
            misplaced_products_count=0,
            fallen_products_count=0,
            reasoning="Error in analysis"
        )

    def _fallback_decision(self, cleanliness_score: float = None) -> AlertDecision:
        """Create fallback decision using simple rules if LLM fails"""
        print("\n⚠️  Using fallback decision logic...")
        
        if cleanliness_score is not None:
            # Make decision based on cleanliness score
            if cleanliness_score >= 8:
                return AlertDecision(
                    alert_required=True,
                    priority=AlertPriority.HIGH,
                    reasoning=f"Cleanliness score is {cleanliness_score}/10, indicating dirty conditions. LLM analysis failed, using rule-based fallback.",
                    recommended_action="Clean the area immediately",
                    estimated_time_minutes=15,
                    confidence_level=0.6
                )
            elif cleanliness_score >= 6:
                return AlertDecision(
                    alert_required=True,
                    priority=AlertPriority.MEDIUM,
                    reasoning=f"Cleanliness score is {cleanliness_score}/10, indicating moderate cleaning needed. LLM analysis failed, using rule-based fallback.",
                    recommended_action="Schedule cleaning soon",
                    estimated_time_minutes=10,
                    confidence_level=0.6
                )
            else:
                return AlertDecision(
                    alert_required=False,
                    priority=AlertPriority.NONE,
                    reasoning=f"Cleanliness score is {cleanliness_score}/10, indicating acceptable conditions. LLM analysis failed, using rule-based fallback.",
                    recommended_action="Continue normal cleaning schedule",
                    estimated_time_minutes=0,
                    confidence_level=0.6
                )
        
        # Default fallback if no score available
        return AlertDecision(
            alert_required=False,
            priority=AlertPriority.NONE,
            reasoning="Error processing LLM response - manual review required",
            recommended_action="Manual inspection needed",
            estimated_time_minutes=0,
            confidence_level=0.0
        )


# ============================================================
# MAIN SYSTEM
# ============================================================

class ImprovedRetailMonitoringSystem:
    """
    Simplified architecture:
    - YOLO: Object detection only
    - Vision LLM: All the intelligent analysis
    """
    
    def __init__(self, api_key: str):
        print("Initializing improved system...")
        self.detector = SimpleYOLODetector()
        self.analyzer = VisionLLMAnalyzer(api_key)
        print("System ready")

    def analyze_cleanliness(
        self,
        image_path: str,
        space_type: SpaceType = SpaceType.AISLE,
        traffic_level: TrafficLevel = TrafficLevel.MEDIUM,
        store_tier: str = "standard",
        hours_since_cleaned: float = 3.0
    ) -> AnalysisResult:
        """Analyze image for cleanliness"""
        
        print(f"\n{'='*60}")
        print(f"CLEANLINESS ANALYSIS: {Path(image_path).name}")
        print(f"{'='*60}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Step 1: YOLO detection
        print("\n[1/3] Running YOLO object detection...")
        detections, width, height = self.detector.detect_objects(image)
        print(f"  Detected {len(detections)} objects")
        
        # Step 2: Vision LLM analysis
        print("\n[2/3] Analyzing image with Vision LLM...")
        cleanliness, spatial = self.analyzer.analyze_cleanliness(
            image=image,
            detections=detections,
            space_type=space_type,
            traffic_level=traffic_level,
            hours_since_cleaned=hours_since_cleaned
        )
        
        print(f"  Cleanliness score: {cleanliness.overall_cleanliness_score}/10")
        print(f"  Floor condition: {cleanliness.floor_condition}")
        print(f"  Floor objects: {len(spatial.floor_objects)}")
        print(f"  Misplaced items: {len(spatial.misplaced_items)}")
        
        # Step 3: Alert decision
        print("\n[3/3] Making alert decision...")
        decision = self.analyzer.make_alert_decision(
            cleanliness=cleanliness,
            spatial=spatial,
            merchandise=None,
            space_type=space_type,
            traffic_level=traffic_level,
            store_tier=store_tier,
            hours_since_cleaned=hours_since_cleaned
        )
        
        self._print_results(cleanliness, spatial, None, decision)
        
        return AnalysisResult(
            cleanliness_analysis=cleanliness,
            spatial_analysis=spatial,
            merchandise_analysis=None,
            alert_decision=decision,
            raw_detections=detections
        )

    def analyze_merchandise(
        self,
        image_path: str,
        expected_shelf_fullness: float = 80.0,
        space_type: SpaceType = SpaceType.AISLE,
        traffic_level: TrafficLevel = TrafficLevel.MEDIUM,
        store_tier: str = "standard"
    ) -> AnalysisResult:
        """Analyze image for merchandise quality"""
        
        print(f"\n{'='*60}")
        print(f"MERCHANDISE ANALYSIS: {Path(image_path).name}")
        print(f"{'='*60}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        print("\n[1/3] Running YOLO object detection...")
        detections, _, _ = self.detector.detect_objects(image)
        print(f"  Detected {len(detections)} objects")
        
        print("\n[2/3] Analyzing merchandise with Vision LLM...")
        merchandise = self.analyzer.analyze_merchandise(
            image=image,
            detections=detections,
            expected_shelf_fullness=expected_shelf_fullness
        )
        
        print(f"  Shelf fullness: {merchandise.shelf_fullness_score}/10")
        print(f"  Organization: {merchandise.shelf_organization_score}/10")
        print(f"  Fallen products: {merchandise.fallen_products_count}")
        
        print("\n[3/3] Making alert decision...")
        decision = self.analyzer.make_alert_decision(
            cleanliness=None,
            spatial=None,
            merchandise=merchandise,
            space_type=space_type,
            traffic_level=traffic_level,
            store_tier=store_tier,
            hours_since_cleaned=0
        )
        
        self._print_results(None, None, merchandise, decision)
        
        return AnalysisResult(
            cleanliness_analysis=None,
            spatial_analysis=None,
            merchandise_analysis=merchandise,
            alert_decision=decision,
            raw_detections=detections
        )

    def _print_results(
        self,
        cleanliness: Optional[CleanlinessAnalysis],
        spatial: Optional[SpatialAnalysis],
        merchandise: Optional[MerchandiseAnalysis],
        decision: AlertDecision
    ):
        """Pretty print results"""
        
        print(f"\n{'='*60}")
        print("ANALYSIS RESULTS")
        print(f"{'='*60}")
        
        if cleanliness:
            print(f"\n🧹 CLEANLINESS:")
            print(f"  Score: {cleanliness.overall_cleanliness_score}/10")
            print(f"  Condition: {cleanliness.floor_condition}")
            if cleanliness.visible_debris:
                print(f"  Debris: {', '.join(cleanliness.visible_debris)}")
            print(f"  Spills: {'Yes' if cleanliness.spills_present else 'No'}")
            print(f"  Stains: {'Yes' if cleanliness.stains_present else 'No'}")
            print(f"\n  Reasoning: {cleanliness.reasoning}")
        
        if spatial:
            print(f"\n📍 SPATIAL ANALYSIS:")
            print(f"  Floor objects: {len(spatial.floor_objects)}")
            print(f"  Shelf objects: {len(spatial.shelf_objects)}")
            print(f"  Misplaced items: {len(spatial.misplaced_items)}")
            if spatial.misplaced_items:
                for item in spatial.misplaced_items[:3]:
                    if isinstance(item, dict):
                        obj = item.get('object', 'unknown')
                        curr = item.get('current_location', '')
                        should = item.get('should_be', '')
                        print(f"    - {obj}: {curr} → {should}")
                    else:
                        print(f"    - {item}")
        
        if merchandise:
            print(f"\n📦 MERCHANDISE:")
            print(f"  Shelf fullness: {merchandise.shelf_fullness_score}/10")
            print(f"  Organization: {merchandise.shelf_organization_score}/10")
            print(f"  Empty spaces: {merchandise.empty_spaces_count}")
            print(f"  Fallen products: {merchandise.fallen_products_count}")
            print(f"\n  Reasoning: {merchandise.reasoning}")
        
        print(f"\n{'='*60}")
        print("🚨 ALERT DECISION")
        print(f"{'='*60}")
        print(f"  Alert Required: {'YES' if decision.alert_required else 'NO'}")
        print(f"  Priority: {decision.priority.value.upper()}")
        print(f"  Confidence: {decision.confidence_level:.0%}")
        print(f"\n  Reasoning: {decision.reasoning}")
        print(f"  Action: {decision.recommended_action}")
        print(f"  Est. Time: {decision.estimated_time_minutes} minutes")
        print(f"{'='*60}\n")


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class VideoProcessor:
    """Process videos frame by frame"""
    
    def __init__(self, system: ImprovedRetailMonitoringSystem):
        self.system = system
    
    def process_video(
        self,
        video_path: str,
        frame_skip: int = 60,
        analysis_type: str = "cleanliness",
        **kwargs
    ):
        """
        Process video by sampling frames.
        Note: This can be expensive with vision LLM calls.
        """
        
        print(f"\n{'='*60}")
        print(f"VIDEO ANALYSIS: {Path(video_path).name}")
        print(f"{'='*60}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"\nVideo info:")
        print(f"  Total frames: {total_frames}")
        print(f"  FPS: {fps}")
        print(f"  Analyzing every {frame_skip} frames")
        
        frame_count = 0
        analyzed_frames = 0
        results = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                print(f"\nAnalyzing frame {frame_count}/{total_frames}...")
                
                # Save frame temporarily
                temp_path = f"/tmp/frame_{frame_count}.jpg"
                cv2.imwrite(temp_path, frame)
                
                try:
                    if analysis_type == "cleanliness":
                        result = self.system.analyze_cleanliness(temp_path, **kwargs)
                    else:
                        result = self.system.analyze_merchandise(temp_path, **kwargs)
                    
                    results.append({
                        "frame": frame_count,
                        "result": result
                    })
                    analyzed_frames += 1
                    
                except Exception as e:
                    print(f"Error analyzing frame {frame_count}: {e}")
                
                # Cleanup
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            frame_count += 1
        
        cap.release()
        
        # Summarize results
        self._summarize_video_results(results, analysis_type)
        
        return results
    
    def _summarize_video_results(self, results: List[Dict], analysis_type: str):
        """Print summary of video analysis"""
        
        if not results:
            print("\nNo frames analyzed")
            return
        
        print(f"\n{'='*60}")
        print("VIDEO ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        alerts = [r for r in results if r["result"].alert_decision.alert_required]
        
        print(f"\nFrames analyzed: {len(results)}")
        print(f"Alerts triggered: {len(alerts)}")
        
        if analysis_type == "cleanliness":
            scores = [r["result"].cleanliness_analysis.overall_cleanliness_score 
                     for r in results if r["result"].cleanliness_analysis]
            if scores:
                print(f"Average cleanliness: {np.mean(scores):.1f}/10")
                print(f"Worst cleanliness: {min(scores):.1f}/10")
        else:
            scores = [r["result"].merchandise_analysis.shelf_fullness_score 
                     for r in results if r["result"].merchandise_analysis]
            if scores:
                print(f"Average shelf fullness: {np.mean(scores):.1f}/10")
        
        if alerts:
            print(f"\nFrames with alerts:")
            for r in alerts[:5]:
                print(f"  Frame {r['frame']}: {r['result'].alert_decision.priority.value} - {r['result'].alert_decision.reasoning[:80]}...")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main execution function for VSCode/local development"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Retail Monitoring System')
    parser.add_argument('--image', type=str, help='Path to single image file')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--folder', type=str, default='media', help='Path to folder with images (default: media)')
    parser.add_argument('--space', type=str, default='aisle', choices=['aisle', 'entrance', 'checkout', 'backroom'], help='Space type')
    parser.add_argument('--traffic', type=str, default='medium', choices=['low', 'medium', 'high'], help='Traffic level')
    parser.add_argument('--tier', type=str, default='standard', help='Store tier (standard/premium/budget)')
    parser.add_argument('--hours-cleaned', type=float, default=3.0, help='Hours since last cleaning')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("IMPROVED RETAIL MONITORING SYSTEM")
    print("="*60)
    
    # Initialize system
    system = ImprovedRetailMonitoringSystem(api_key=GEMINI_API_KEY)
    
    # Map string arguments to enums
    space_type = SpaceType[args.space.upper()]
    traffic_level = TrafficLevel[args.traffic.upper()]
    
    # Determine what to process
    if args.image:
        # Single image
        print(f"\nProcessing image: {args.image}")
        uploaded_files = [args.image]
    elif args.video:
        # Single video
        print(f"\nProcessing video: {args.video}")
        uploaded_files = [args.video]
    else:
        # Folder of images/videos
        print(f"\nScanning folder: {args.folder}")
        if not os.path.exists(args.folder):
            os.makedirs(args.folder)
            print(f"Created folder: {args.folder}")
            print("Please add images/videos to this folder and run again")
            return
        
        # Get all image and video files
        uploaded_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.mp4', '*.avi', '*.mov', '*.mkv']:
            uploaded_files.extend(list(Path(args.folder).glob(ext)))
        
        if not uploaded_files:
            print(f"No images or videos found in {args.folder}/")
            print("Supported formats: jpg, jpeg, png, bmp, mp4, avi, mov, mkv")
            return
        
        print(f"Found {len(uploaded_files)} file(s)")
    
    # Process files
    video_processor = VideoProcessor(system)
    
    for file_path in uploaded_files:
        file_path = str(file_path)
        
        if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # Video processing
            video_processor.process_video(
                video_path=file_path,
                frame_skip=60,
                analysis_type="cleanliness",
                space_type=SpaceType.AISLE,
                traffic_level=TrafficLevel.MEDIUM,
                store_tier="standard",
                hours_since_cleaned=3.0
            )
        else:
            # Image analysis
            system.analyze_cleanliness(
                image_path=file_path,
                space_type=SpaceType.AISLE,
                traffic_level=TrafficLevel.MEDIUM,
                store_tier="standard",
                hours_since_cleaned=3.0
            )


if __name__ == "__main__":
    main()