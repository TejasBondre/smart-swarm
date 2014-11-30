var numBoids = prompt("Enter the number of agents (around 100).");
var dim = 100;
var cohesionDist = 20;
var cohesionRatio = 0.001;
var separationDist = 5;
var separationRatio = -0.0005;
var firstTime = 0;
var boids = [];

function Boid () {
	this.pos = new THREE.Vector3( (Math.random()*dim)-(dim/2), (Math.random()*dim)-(dim/2), (Math.random()*dim)-(dim/2));
	this.vel = new THREE.Vector3( Math.random()-0.5, Math.random()-0.5, Math.random()-0.5 );
	this.vel.x = this.vel.x/10; this.vel.y = this.vel.y/10; this.vel.z = this.vel.z/10; 
	this.mat = new THREE.PointCloudMaterial( { color: new THREE.Color( Math.random(), Math.random(), Math.random() ) } );
};

var updatePos = function(k) {
	var numNeighbours1=0; var numNeighbours2=0;
	var center1 = new THREE.Vector3(0,0,0);
	var center2 = new THREE.Vector3(0,0,0);
	for (var i = 0; i< numBoids; i++) {
		var magnitude = ((boids[i].pos.x - boids[k].pos.x)*(boids[i].pos.x - boids[k].pos.x)) +
							 ((boids[i].pos.y - boids[k].pos.y)*(boids[i].pos.y - boids[k].pos.y)) +
							 ((boids[i].pos.z - boids[k].pos.z)*(boids[i].pos.z - boids[k].pos.z));
		if (magnitude < cohesionDist*cohesionDist) {
			center1.x = center1.x + boids[i].pos.x;
			center1.y = center1.y + boids[i].pos.y;
			center1.z = center1.z + boids[i].pos.z;
			numNeighbours1++;
		}
		if (magnitude < separationDist*separationDist) {
			center2.x = center2.x + boids[i].pos.x;
			center2.y = center2.y + boids[i].pos.y;
			center2.z = center2.z + boids[i].pos.z;
			numNeighbours2++;
		}
	}
	center1.x = center1.x / numNeighbours1; center2.x = center2.x / numNeighbours2;
	center1.y = center1.y / numNeighbours1; center2.y = center2.y / numNeighbours2;
	center1.z = center1.z / numNeighbours1; center2.z = center2.z / numNeighbours2;
	boids[k].vel.x = boids[k].vel.x + ((center1.x - boids[k].pos.x) * cohesionRatio) + ((center2.x - boids[k].pos.x) * separationRatio);
	boids[k].vel.y = boids[k].vel.y + ((center1.y - boids[k].pos.y) * cohesionRatio) + ((center2.y - boids[k].pos.y) * separationRatio);
	boids[k].vel.z = boids[k].vel.z + ((center1.z - boids[k].pos.z) * cohesionRatio) + ((center2.z - boids[k].pos.z) * separationRatio);
	boids[k].pos = boids[k].pos.add(boids[k].vel);
	if (boids[k].pos.x > dim/2) boids[k].pos.x = boids[k].pos.x - dim; if (boids[k].pos.x < (0-dim/2)) boids[k].pos.x = boids[k].pos.x + dim;
	if (boids[k].pos.y > dim/2) boids[k].pos.y = boids[k].pos.y - dim; if (boids[k].pos.y < (0-dim/2)) boids[k].pos.y = boids[k].pos.y + dim;
	if (boids[k].pos.z > dim/2) boids[k].pos.z = boids[k].pos.z - dim; if (boids[k].pos.z < (0-dim/2)) boids[k].pos.z = boids[k].pos.z + dim;
};

var camera = new THREE.PerspectiveCamera( 75, window.innerWidth/window.innerHeight, 0.1, 1000 );
var renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth, window.innerHeight );
document.body.appendChild( renderer.domElement );
camera.position.z = dim/2;
for (var i=0; i<numBoids; i++) boids[i] = new Boid();

var render = function () {
	var scene = new THREE.Scene();
	for (var i=0; i<numBoids; i++) {
		updatePos(i);
		var geometry = new THREE.Geometry();
		geometry.vertices.push( boids[i].pos);
		scene.add( new THREE.PointCloud(geometry, boids[i].mat) );
	}
	requestAnimationFrame( render );
	renderer.render( scene, camera );
};

render();